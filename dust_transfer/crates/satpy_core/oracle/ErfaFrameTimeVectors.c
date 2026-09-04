/*
 * Task 4AF offline refinement probe.
 *
 * ERFA remains the UTC/TAI/TT, X/Y/s, RC2I, and RPOM authority. This file
 * raises only EOP interpolation, ERA, outer matrix composition, and the
 * conditioned five-point stencil to generator-local double-double arithmetic.
 * It emits no accepted fixture before the exact-constant checkpoint.
 */
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "erfa.h"
#include "erfam.h"

extern const char *eraVersion(void);
extern const char *eraSofaVersion(void);

typedef struct { double hi, lo; } dd;
typedef struct { double xp, yp, dut1, dx, dy; } eop;
typedef struct {
    int y, m, d, hh, mm;
    double ss;
    const char *name;
} epoch;
typedef struct {
    int y, m, d;
    double jd, offset, reference_mjd, slope;
} tai_utc_entry;

static dd dd_make(double hi, double lo) { dd x = {hi, lo}; return x; }
static dd dd_from(double x) { return dd_make(x, 0.0); }
static dd dd_normalize(double a, double b) {
    const double s = a + b;
    const double e = b - (s - a);
    return dd_make(s, e);
}
static dd dd_add(dd a, dd b) {
    const double s = a.hi + b.hi;
    const double v = s - a.hi;
    const double e = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    return dd_normalize(s, e);
}
static dd dd_neg(dd a) { return dd_make(-a.hi, -a.lo); }
static dd dd_sub(dd a, dd b) { return dd_add(a, dd_neg(b)); }
static dd dd_mul(dd a, dd b) {
    const double p = a.hi * b.hi;
    const double e = fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi;
    return dd_normalize(p, e);
}
static dd dd_scale(dd a, double b) { return dd_mul(a, dd_from(b)); }
static dd dd_div(dd a, dd b) {
    const double q1 = a.hi / b.hi;
    const dd remainder = dd_sub(a, dd_mul(b, dd_from(q1)));
    const double q2 = (remainder.hi + remainder.lo) / b.hi;
    return dd_add(dd_from(q1), dd_from(q2));
}
static double dd_to_double(dd a) { return a.hi + a.lo; }

static dd lagrange(dd x, const dd xs[4], const double ys[4]) {
    dd sum = dd_from(0.0);
    int i;
    for (i = 0; i < 4; ++i) {
        dd term = dd_from(ys[i]);
        int j;
        for (j = 0; j < 4; ++j) {
            if (j != i) {
                term = dd_mul(term, dd_div(dd_sub(x, xs[j]), dd_sub(xs[i], xs[j])));
            }
        }
        sum = dd_add(sum, term);
    }
    return sum;
}

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static double fixed_field(const char *line, int start, int width) {
    char buffer[32];
    char *end;
    double value;
    memcpy(buffer, line + start, (size_t)width);
    buffer[width] = '\0';
    value = strtod(buffer, &end);
    if (end == buffer || !isfinite(value)) fail("invalid finite EOP field");
    return value;
}

static eop load_eop(const char *path, int wanted_mjd) {
    FILE *input = fopen(path, "r");
    char line[256];
    int matches = 0;
    eop result = {0, 0, 0, 0, 0};
    if (input == NULL) fail("cannot open finals2000A.all");
    while (fgets(line, sizeof(line), input) != NULL) {
        const size_t length = strlen(line);
        int mjd;
        if (length < 125) fail("short finals2000A record");
        mjd = (int)fixed_field(line, 7, 8);
        if (mjd == wanted_mjd) {
            if ((line[16] != 'I' && line[16] != 'P') ||
                (line[57] != 'I' && line[57] != 'P') ||
                (line[95] != 'I' && line[95] != 'P')) {
                fail("invalid selected Bulletin-A flag");
            }
            result.xp = fixed_field(line, 18, 9);
            result.yp = fixed_field(line, 37, 9);
            result.dut1 = fixed_field(line, 58, 10);
            result.dx = fixed_field(line, 97, 9);
            result.dy = fixed_field(line, 116, 9);
            ++matches;
        }
    }
    if (fclose(input) != 0) fail("cannot close finals2000A.all");
    if (matches != 1) fail("selected EOP MJD must occur exactly once");
    return result;
}

static void anchor_dates(const epoch *ep, double *utc1, double *utc2,
                         double *tai1, double *tai2) {
    if (eraDtf2d("UTC", ep->y, ep->m, ep->d, ep->hh, ep->mm, ep->ss,
                 utc1, utc2) < 0 ||
        eraUtctai(*utc1, *utc2, tai1, tai2) != 0) {
        fail("anchor UTC/TAI conversion failed");
    }
}

static int civil_mjd(const epoch *ep) {
    double zero, mjd;
    if (eraCal2jd(ep->y, ep->m, ep->d, &zero, &mjd) != 0 ||
        zero != 2400000.5 || mjd != floor(mjd)) {
        fail("anchor civil MJD conversion failed");
    }
    return (int)mjd;
}

static int month_number(const char *month) {
    static const char *names[] = {
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
    };
    int index;
    for (index = 0; index < 12; ++index) {
        if (strcmp(month, names[index]) == 0) return index + 1;
    }
    fail("invalid tai-utc.dat month");
    return 0;
}

static int date_not_after(int y, int m, int d, const epoch *ep) {
    if (y != ep->y) return y < ep->y;
    if (m != ep->m) return m < ep->m;
    return d <= ep->d;
}

static void validate_tai_utc(const char *path, const epoch *epochs,
                             size_t epoch_count) {
    FILE *input = fopen(path, "r");
    tai_utc_entry entries[64];
    char line[256];
    size_t count = 0, post_1972_count = 0, e;
    if (input == NULL) fail("cannot open tai-utc.dat");
    while (fgets(line, sizeof(line), input) != NULL) {
        char month[4];
        int consumed = 0;
        double zero, mjd, erfa_dat, expected;
        tai_utc_entry *entry;
        const char *tail;
        if (count == sizeof(entries) / sizeof(entries[0])) {
            fail("too many tai-utc.dat records");
        }
        entry = &entries[count];
        if (sscanf(line,
                   " %d %3s %d =JD %lf TAI-UTC= %lf S + (MJD - %lf) X %lf S %n",
                   &entry->y, month, &entry->d, &entry->jd, &entry->offset,
                   &entry->reference_mjd, &entry->slope, &consumed) != 7) {
            fail("invalid tai-utc.dat record");
        }
        for (tail = line + consumed; *tail != '\0'; ++tail) {
            if (*tail != ' ' && *tail != '\t' && *tail != '\r' && *tail != '\n') {
                fail("trailing tai-utc.dat record content");
            }
        }
        entry->m = month_number(month);
        if (eraCal2jd(entry->y, entry->m, entry->d, &zero, &mjd) != 0 ||
            fabs((zero + mjd) - entry->jd) > 1e-12 ||
            (count != 0 && entry->jd <= entries[count - 1].jd)) {
            fail("invalid tai-utc.dat transition date");
        }
        if (entry->y >= 1972) {
            expected =
                entry->offset + (mjd - entry->reference_mjd) * entry->slope;
            if (eraDat(entry->y, entry->m, entry->d, 0.0, &erfa_dat) < 0 ||
                fabs(erfa_dat - expected) > 1e-12) {
                fail("tai-utc.dat transition disagrees with ERFA eraDat");
            }
            ++post_1972_count;
        }
        ++count;
    }
    if (ferror(input)) fail("cannot read tai-utc.dat");
    if (fclose(input) != 0) fail("cannot close tai-utc.dat");
    if (count != 41 || post_1972_count != 28) {
        fail("tai-utc.dat record count mismatch");
    }
    for (e = 0; e < epoch_count; ++e) {
        const epoch *ep = &epochs[e];
        const tai_utc_entry *active = NULL;
        double zero, mjd, day_fraction, expected, erfa_dat;
        size_t index;
        for (index = 0; index < count; ++index) {
            if (date_not_after(entries[index].y, entries[index].m,
                               entries[index].d, ep)) {
                active = &entries[index];
            }
        }
        if (active == NULL || active->y < 1972 ||
            eraCal2jd(ep->y, ep->m, ep->d, &zero, &mjd) != 0) {
            fail("corpus epoch lacks post-1972 tai-utc.dat coverage");
        }
        day_fraction =
            (ep->hh * 3600.0 + ep->mm * 60.0 + fmin(ep->ss, 59.999999)) /
            ERFA_DAYSEC;
        expected = active->offset +
                   (mjd + day_fraction - active->reference_mjd) *
                       active->slope;
        if (eraDat(ep->y, ep->m, ep->d, day_fraction, &erfa_dat) < 0 ||
            fabs(erfa_dat - expected) > 1e-12) {
            fail("corpus tai-utc.dat value disagrees with ERFA eraDat");
        }
    }
}

static double dat_at_mjd(int mjd) {
    int y, m, d;
    double fraction, dat;
    if (eraJd2cal(2400000.5, (double)mjd, &y, &m, &d, &fraction) != 0 ||
        eraDat(y, m, d, 0.0, &dat) < 0) {
        fail("node TAI-UTC conversion failed");
    }
    return dat;
}

static dd dd_era(dd ut11, dd ut12) {
    dd d1 = dd_to_double(ut11) < dd_to_double(ut12) ? ut11 : ut12;
    dd d2 = dd_to_double(ut11) < dd_to_double(ut12) ? ut12 : ut11;
    dd t = dd_add(d1, dd_sub(d2, dd_from(2451545.0)));
    dd f = dd_add(dd_sub(d1, dd_from(floor(d1.hi))),
                  dd_sub(d2, dd_from(floor(d2.hi))));
    const dd c0 = {0x1.8ee0984cc2772p-1, 0x1.37791827271b7p-56};
    const dd c1 = {0x1.66d9b93e65515p-9, 0x1.1a9fd4c729390p-63};
    const dd tau = {0x1.921fb54442d18p+2, 0x1.1a62633145c07p-52};
    dd theta = dd_mul(tau, dd_add(dd_add(f, c0), dd_mul(c1, t)));
    const double turns = floor(dd_to_double(dd_div(theta, tau)));
    theta = dd_sub(theta, dd_scale(tau, turns));
    if (dd_to_double(theta) < 0.0) theta = dd_add(theta, tau);
    return theta;
}

static void dd_sincos(dd x, dd *sine, dd *cosine) {
    const double s = sin(x.hi);
    const double c = cos(x.hi);
    *sine = dd_add(dd_from(s), dd_from(c * x.lo));
    *cosine = dd_sub(dd_from(c), dd_from(s * x.lo));
}

static void matrix_multiply(const dd a[3][3], const dd b[3][3], dd out[3][3]) {
    dd temporary[3][3];
    int i, j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            dd value = dd_from(0.0);
            int k;
            for (k = 0; k < 3; ++k) value = dd_add(value, dd_mul(a[i][k], b[k][j]));
            temporary[i][j] = value;
        }
    }
    memcpy(out, temporary, sizeof(temporary));
}

static void crosscheck_ut1(dd sample1, dd sample2, dd raw_ut1_tai) {
    double utc1, utc2, dat, direct1, direct2, reconstructed1, reconstructed2;
    double fraction;
    int y, m, d;
    const double tai1 = dd_to_double(sample1);
    const double tai2 = dd_to_double(sample2);
    const double raw = dd_to_double(raw_ut1_tai);
    dd difference;
    if (eraTaiutc(tai1, tai2, &utc1, &utc2) < 0 ||
        eraJd2cal(utc1, utc2, &y, &m, &d, &fraction) != 0 ||
        eraDat(y, m, d, 0.0, &dat) < 0 ||
        eraTaiut1(tai1, tai2, raw, &direct1, &direct2) != 0 ||
        eraUtcut1(utc1, utc2, raw + dat, &reconstructed1, &reconstructed2) < 0) {
        fail("UT1 cross-check conversion failed");
    }
    difference = dd_add(dd_sub(dd_from(direct1), dd_from(reconstructed1)),
                        dd_sub(dd_from(direct2), dd_from(reconstructed2)));
    if (fabs(dd_to_double(dd_scale(difference, ERFA_DAYSEC))) > 1e-9) {
        fail("eraTaiut1/eraUtcut1 cross-check exceeds 1 ns");
    }
}

static void frame_matrix(const char *path, const epoch *ep, int real_eop,
                         double offset, dd out[3][3], double ordinary[3][3]) {
    double utc1, utc2, anchor1, anchor2;
    dd sample1, sample2, raw;
    double xp = 0.0, yp = 0.0, dx = 0.0, dy = 0.0;
    double tt1, tt2, x, y, s, rc2i[3][3], rpom[3][3];
    dd r3[3][3], rc2i_dd[3][3], rpom_dd[3][3], intermediate[3][3];
    int i, j;
    anchor_dates(ep, &utc1, &utc2, &anchor1, &anchor2);
    sample1 = dd_from(anchor1);
    sample2 = dd_add(dd_from(anchor2), dd_from(offset / ERFA_DAYSEC));

    if (real_eop) {
        const int center = civil_mjd(ep);
        dd abscissae[4];
        double raw_values[4], xp_values[4], yp_values[4], dx_values[4], dy_values[4];
        int q;
        for (q = 0; q < 4; ++q) {
            const int mjd = center - 1 + q;
            const eop value = load_eop(path, mjd);
            double node1, node2;
            if (eraUtctai(2400000.5, (double)mjd, &node1, &node2) != 0) {
                fail("EOP node UTC/TAI conversion failed");
            }
            abscissae[q] =
                dd_scale(dd_add(dd_sub(dd_from(node1), dd_from(anchor1)),
                                dd_sub(dd_from(node2), dd_from(anchor2))),
                         ERFA_DAYSEC);
            raw_values[q] = value.dut1 - dat_at_mjd(mjd);
            xp_values[q] = value.xp;
            yp_values[q] = value.yp;
            dx_values[q] = value.dx;
            dy_values[q] = value.dy;
        }
        raw = lagrange(dd_from(offset), abscissae, raw_values);
        xp = dd_to_double(lagrange(dd_from(offset), abscissae, xp_values)) * ERFA_DAS2R;
        yp = dd_to_double(lagrange(dd_from(offset), abscissae, yp_values)) * ERFA_DAS2R;
        dx = dd_to_double(lagrange(dd_from(offset), abscissae, dx_values)) * 1e-3 * ERFA_DAS2R;
        dy = dd_to_double(lagrange(dd_from(offset), abscissae, dy_values)) * 1e-3 * ERFA_DAS2R;
    } else {
        double dat;
        const double day_fraction =
            (ep->hh * 3600.0 + ep->mm * 60.0 + fmin(ep->ss, 59.999999)) / ERFA_DAYSEC;
        if (eraDat(ep->y, ep->m, ep->d, day_fraction, &dat) < 0) {
            fail("zero-EOP anchor TAI-UTC conversion failed");
        }
        raw = dd_from(-dat);
    }

    crosscheck_ut1(sample1, sample2, raw);
    if (eraTaitt(dd_to_double(sample1), dd_to_double(sample2), &tt1, &tt2) != 0) {
        fail("TAI/TT conversion failed");
    }
    eraXys06a(tt1, tt2, &x, &y, &s);
    eraC2ixys(x + dx, y + dy, s, rc2i);
    eraPom00(xp, yp, eraSp00(tt1, tt2), rpom);

    {
        const dd ut11 = sample1;
        const dd ut12 = dd_add(sample2, dd_div(raw, dd_from(ERFA_DAYSEC)));
        dd sine, cosine;
        dd_sincos(dd_era(ut11, ut12), &sine, &cosine);
        r3[0][0] = cosine; r3[0][1] = sine; r3[0][2] = dd_from(0.0);
        r3[1][0] = dd_neg(sine); r3[1][1] = cosine; r3[1][2] = dd_from(0.0);
        r3[2][0] = dd_from(0.0); r3[2][1] = dd_from(0.0); r3[2][2] = dd_from(1.0);
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                rc2i_dd[i][j] = dd_from(rc2i[i][j]);
                rpom_dd[i][j] = dd_from(rpom[i][j]);
            }
        }
        matrix_multiply(r3, rc2i_dd, intermediate);
        matrix_multiply(rpom_dd, intermediate, out);
        eraC2tcio(rc2i, eraEra00(dd_to_double(ut11), dd_to_double(ut12)),
                  rpom, ordinary);
    }
}

static void derivatives(const char *path, const epoch *ep, int real_eop, double h,
                        dd value[3][3], dd derivative[3][3], dd second[3][3],
                        double *r_error) {
    dd samples[5][3][3];
    double ordinary[3][3];
    int q, i, j;
    *r_error = 0.0;
    for (q = -2; q <= 2; ++q) {
        frame_matrix(path, ep, real_eop, q * h, samples[q + 2], ordinary);
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                const double error = fabs(dd_to_double(samples[q + 2][i][j]) - ordinary[i][j]);
                if (error > *r_error) *r_error = error;
            }
        }
    }
    memcpy(value, samples[2], sizeof(samples[2]));
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            const dd near_difference = dd_sub(samples[3][i][j], samples[1][i][j]);
            const dd far_difference = dd_sub(samples[4][i][j], samples[0][i][j]);
            const dd near_second =
                dd_add(dd_sub(samples[3][i][j], samples[2][i][j]),
                       dd_sub(samples[1][i][j], samples[2][i][j]));
            const dd far_second =
                dd_add(dd_sub(samples[4][i][j], samples[2][i][j]),
                       dd_sub(samples[0][i][j], samples[2][i][j]));
            derivative[i][j] =
                dd_div(dd_sub(dd_scale(near_difference, 8.0), far_difference),
                       dd_from(12.0 * h));
            second[i][j] =
                dd_div(dd_sub(dd_scale(near_second, 16.0), far_second),
                       dd_from(12.0 * h * h));
        }
    }
}

static double maximum_difference(dd a[3][3], dd b[3][3]) {
    double maximum = 0.0;
    int i, j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            const double error = fabs(dd_to_double(a[i][j]) - dd_to_double(b[i][j]));
            if (error > maximum) maximum = error;
        }
    }
    return maximum;
}

static void refinement_bounds(const char *path, const epoch *epochs,
                              size_t epoch_count, double *max_r,
                              double *max_rdot, double *max_rddot) {
    size_t e;
    int real_eop;
    *max_r = 0.0;
    *max_rdot = 0.0;
    *max_rddot = 0.0;
    for (e = 0; e < epoch_count; ++e) {
        for (real_eop = 0; real_eop <= 1; ++real_eop) {
            dd value_half[3][3], derivative_half[3][3], second_half[3][3];
            dd value_quarter[3][3], derivative_quarter[3][3], second_quarter[3][3];
            double r_half, r_quarter;
            double rdot_error;
            derivatives(path, &epochs[e], real_eop, 0.5, value_half,
                        derivative_half, second_half, &r_half);
            derivatives(path, &epochs[e], real_eop, 0.25, value_quarter,
                        derivative_quarter, second_quarter, &r_quarter);
            rdot_error = maximum_difference(derivative_half, derivative_quarter);
            const double rddot_error =
                maximum_difference(second_half, second_quarter);
            (void)r_half;
            if (r_quarter > *max_r) *max_r = r_quarter;
            if (rdot_error > *max_rdot) *max_rdot = rdot_error;
            if (rddot_error > *max_rddot) *max_rddot = rddot_error;
        }
    }
}

static void print_hex(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    printf("\"0x%016" PRIx64 "\"", bits);
}

static void print_vector(const double value[3]) {
    int i;
    putchar('[');
    for (i = 0; i < 3; ++i) {
        if (i != 0) putchar(',');
        print_hex(value[i]);
    }
    putchar(']');
}

static void print_matrix(const dd value[3][3]) {
    int i, j;
    putchar('[');
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            if (i != 0 || j != 0) putchar(',');
            print_hex(dd_to_double(value[i][j]));
        }
    }
    putchar(']');
}

static void transform_state(const dd r[3][3], const dd rdot[3][3],
                            const dd rddot[3][3], const double inertial_r[3],
                            const double inertial_v[3], const double inertial_a[3],
                            double terrestrial_r[3], double terrestrial_v[3],
                            double terrestrial_a[3]) {
    int i, j;
    for (i = 0; i < 3; ++i) {
        dd position = dd_from(0.0);
        dd velocity = dd_from(0.0);
        dd acceleration = dd_from(0.0);
        for (j = 0; j < 3; ++j) {
            position = dd_add(position, dd_scale(r[i][j], inertial_r[j]));
            velocity = dd_add(
                velocity,
                dd_add(dd_scale(r[i][j], inertial_v[j]),
                       dd_scale(rdot[i][j], inertial_r[j])));
            acceleration = dd_add(
                acceleration,
                dd_add(dd_scale(r[i][j], inertial_a[j]),
                       dd_add(dd_scale(rdot[i][j], 2.0 * inertial_v[j]),
                              dd_scale(rddot[i][j], inertial_r[j]))));
        }
        terrestrial_r[i] = dd_to_double(position);
        terrestrial_v[i] = dd_to_double(velocity);
        terrestrial_a[i] = dd_to_double(acceleration);
        if (!isfinite(terrestrial_r[i]) || !isfinite(terrestrial_v[i]) ||
            !isfinite(terrestrial_a[i])) {
            fail("nonfinite transformed state");
        }
    }
}

static void print_candidate(const char *path, const epoch *epochs,
                            size_t epoch_count, double max_r, double max_rdot,
                            double max_rddot) {
    static const double positions[2][3] = {
        {7000.0, 0.0, 0.0},
        {0.0, -7000.0, 1000.0}
    };
    static const double velocities[2][3] = {
        {0.0, 7.5, 1.0},
        {-7.4, 0.2, 0.0}
    };
    static const double accelerations[2][3] = {
        {-0.008, 0.0, 0.0},
        {0.0, 0.0075, -0.001}
    };
    size_t e;
    int real_eop, state_index;
    int case_index = 0;
    printf("{\"schema\":\"part_a_erfa_sofa_derived_frame_time_v1\","
           "\"semantic_sha256\":\"__SEMANTIC_SHA256__\","
           "\"authority_label\":\"ERFA 2.0.1 / SOFA 20231011-derived\","
           "\"claim_scope\":\"Offline GCRS-to-ITRS characterization source "
           "output only; production comparison deferred\","
           "\"provenance\":{\"generator_source_sha256\":"
           "\"__GENERATOR_SOURCE_SHA256__\",\"orchestration_script_sha256\":"
           "\"__ORCHESTRATION_SCRIPT_SHA256__\",\"frame_time_manifest_sha256\":"
           "\"__FRAME_TIME_MANIFEST_SHA256__\","
           "\"frame_time_manifest_semantic_sha256\":"
           "\"__FRAME_TIME_MANIFEST_SEMANTIC_SHA256__\","
           "\"erfa_source_archive_sha256\":\"__ERFA_SOURCE_ARCHIVE_SHA256__\","
           "\"erfa_source_aggregate_sha256\":"
           "\"__ERFA_SOURCE_AGGREGATE_SHA256__\","
           "\"finals2000a_sha256\":\"__FINALS2000A_SHA256__\","
           "\"tai_utc_sha256\":\"__TAI_UTC_SHA256__\","
           "\"erfa_version\":\"2.0.1\",\"sofa_version\":\"20231011\"},"
           "\"toolchain\":{\"architecture\":\"__ARCHITECTURE__\","
           "\"sw_vers\":\"__SW_VERS__\",\"clang_path\":\"__CLANG_PATH__\","
           "\"clang_sha256\":\"__CLANG_SHA256__\",\"sdk_path\":\"__SDK_PATH__\","
           "\"libsystem_tbd_sha256\":\"__LIBSYSTEM_TBD_SHA256__\","
           "\"dyld_cache_main_sha256\":\"__DYLD_CACHE_MAIN_SHA256__\","
           "\"dyld_cache_atlas_sha256\":\"__DYLD_CACHE_ATLAS_SHA256__\","
           "\"dyld_cache_map_sha256\":\"__DYLD_CACHE_MAP_SHA256__\","
           "\"compile_argv\":\"__COMPILE_ARGV__\","
           "\"otool_l\":\"__OTOOL_L__\","
           "\"generator_binary_sha256\":\"__GENERATOR_BINARY_SHA256__\"},"
           "\"canonicalization\":{\"semantic_domain\":"
           "\"PART_A_FRAME_TIME_ORACLE_V1\","
           "\"numeric_encoding\":\"lowercase hexadecimal prefix plus 16 "
           "binary64 digits\","
           "\"matrix_layout\":\"row-major\","
           "\"case_order\":\"epoch,zero_then_real_eop,state_index\","
           "\"json\":\"source-order minified UTF-8 one LF; semantic hash omits "
           "only root semantic_sha256\"},"
           "\"time_and_frame\":{\"input_frame\":\"GCRS\","
           "\"output_frame\":\"ITRS\",\"rotation\":\"GCRS-to-ITRS\","
           "\"anchor_time_scale\":\"UTC via eraDtf2d then continuous TAI\","
           "\"tt\":\"ERFA eraTaitt\","
           "\"real_eop\":\"four-node anchor-local continuous-TAI Lagrange\","
           "\"zero_eop\":\"xp=yp=dX=dY=UT1-UTC=0 at anchor\","
           "\"derivatives\":\"conditioned centered five-point h=0.25 s\"},"
           "\"units\":{\"r\":\"km\",\"v\":\"km/s\",\"a\":\"km/s^2\","
           "\"R\":\"1\",\"Rdot\":\"1/s\",\"Rddot\":\"1/s^2\"},"
           "\"refinement\":{\"h_fixture_s\":");
    print_hex(0.25);
    printf(",\"h_comparison_s\":");
    print_hex(0.5);
    printf(",\"max_r_vs_erfa\":");
    print_hex(max_r);
    printf(",\"max_rdot_difference_s1\":");
    print_hex(max_rdot);
    printf(",\"max_rddot_difference_s2\":");
    print_hex(max_rddot);
    printf("},\"cases\":[");
    for (e = 0; e < epoch_count; ++e) {
        for (real_eop = 0; real_eop <= 1; ++real_eop) {
            dd r[3][3], rdot[3][3], rddot[3][3];
            double r_error;
            derivatives(path, &epochs[e], real_eop, 0.25, r, rdot, rddot,
                        &r_error);
            for (state_index = 0; state_index < 2; ++state_index) {
                double terrestrial_r[3], terrestrial_v[3], terrestrial_a[3];
                transform_state(r, rdot, rddot, positions[state_index],
                                velocities[state_index],
                                accelerations[state_index], terrestrial_r,
                                terrestrial_v, terrestrial_a);
                if (case_index++ != 0) putchar(',');
                printf("{\"id\":\"%s_%s_state_%d\",\"epoch_utc\":\"%s\","
                       "\"eop_policy\":\"%s\",\"state_index\":%d,"
                       "\"r_gcrs_km\":",
                       epochs[e].name, real_eop ? "real_eop" : "zero_eop",
                       state_index, epochs[e].name,
                       real_eop ? "real_eop" : "zero_eop", state_index);
                print_vector(positions[state_index]);
                printf(",\"v_gcrs_km_s\":");
                print_vector(velocities[state_index]);
                printf(",\"a_gcrs_km_s2\":");
                print_vector(accelerations[state_index]);
                printf(",\"R_gcrs_to_itrs\":");
                print_matrix(r);
                printf(",\"Rdot_s1\":");
                print_matrix(rdot);
                printf(",\"Rddot_s2\":");
                print_matrix(rddot);
                printf(",\"r_itrs_km\":");
                print_vector(terrestrial_r);
                printf(",\"v_itrs_km_s\":");
                print_vector(terrestrial_v);
                printf(",\"a_itrs_km_s2\":");
                print_vector(terrestrial_a);
                putchar('}');
            }
        }
    }
    printf("]}\n");
    if (case_index != 20) fail("candidate case count mismatch");
}

int main(int argc, char **argv) {
    static const epoch epochs[] = {
        {2000, 1, 1, 12, 0, 0.0, "2000-01-01T12:00:00"},
        {2016, 12, 31, 23, 59, 59.0, "2016-12-31T23:59:59"},
        {2016, 12, 31, 23, 59, 60.0, "2016-12-31T23:59:60"},
        {2017, 1, 1, 0, 0, 0.0, "2017-01-01T00:00:00"},
        {2024, 1, 1, 0, 0, 0.0, "2024-01-01T00:00:00"}
    };
    double max_r, max_rdot, max_rddot;
    int candidate;
    if (argc != 4) {
        fail("usage: ErfaFrameTimeVectors "
             "--refinement-probe|--unsealed-candidate FINALS2000A TAI_UTC");
    }
    candidate = strcmp(argv[1], "--unsealed-candidate") == 0;
    if (!candidate && strcmp(argv[1], "--refinement-probe") != 0) {
        fail("usage: ErfaFrameTimeVectors "
             "--refinement-probe|--unsealed-candidate FINALS2000A TAI_UTC");
    }
    if (FLT_RADIX != 2 || DBL_MANT_DIG != 53 || sizeof(double) != 8 ||
        fegetround() != FE_TONEAREST) {
        fail("refinement probe requires IEEE binary64 round-to-nearest");
    }
    if (strcmp(eraVersion(), "2.0.1") != 0 ||
        strcmp(eraSofaVersion(), "20231011") != 0) {
        fail("compiled ERFA/SOFA-derived version mismatch");
    }
    validate_tai_utc(argv[3], epochs, sizeof(epochs) / sizeof(epochs[0]));
    refinement_bounds(argv[2], epochs, sizeof(epochs) / sizeof(epochs[0]),
                      &max_r, &max_rdot, &max_rddot);
    if (max_r > 5e-13 || max_rdot > 1e-13 || max_rddot > 1e-12) {
        fail("frame/time refinement bound failed");
    }
    if (candidate) {
        print_candidate(argv[2], epochs, sizeof(epochs) / sizeof(epochs[0]),
                        max_r, max_rdot, max_rddot);
    } else {
        printf("max_r=%.17e\nmax_rdot=%.17e\nmax_rddot=%.17e\n",
               max_r, max_rdot, max_rddot);
    }
    return 0;
}
