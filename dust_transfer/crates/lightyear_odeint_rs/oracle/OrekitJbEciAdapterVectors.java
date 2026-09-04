import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.Locale;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.geometry.euclidean.threed.FieldRotation;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.FieldTransform;
import org.orekit.frames.Frame;
import org.orekit.frames.Transform;
import org.orekit.frames.TransformProvider;
import org.orekit.models.earth.atmosphere.JB2008;
import org.orekit.models.earth.atmosphere.JB2008InputParameters;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.ConstantOffsetTimeScale;
import org.orekit.time.DateTimeComponents;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.time.TimeOffset;
import org.orekit.time.TimeScale;
import org.orekit.utils.Constants;
import org.orekit.utils.ExtendedPositionProvider;

/** Data-free Orekit JB2008 synthetic mapping-oracle generator. */
public strictfp final class OrekitJbEciAdapterVectors {

    private static final String SCHEMA =
        "part_a_orekit_jb2008_synthetic_adapter_v1";
    private static final String AUTHORITY =
        "Orekit 13.1.2 synthetic-frame JB2008 mapping oracle";
    private static final String CLAIM =
        "Orekit synthetic-frame mapping oracle only; production Rust adapter comparison deferred";
    private static final String HASH_DOMAIN =
        "PART_A_OREKIT_JB2008_SYNTHETIC_ADAPTER_V1";
    private static final String HASH_ALGORITHM =
        "sha256(domain_ascii || NUL || big_endian_u64_payload_length || canonical_json_without_semantic_sha256)";

    private static final String OREKIT_SHA256 =
        "89c2060c60dbe194a87dddcf3bb8343ebd16733958efe4dcc996cebbbeed655d";
    private static final String HIP_CORE_SHA256 =
        "7c56992f3af64429d871c33c00808ee5db5d9ed56b395b5d3d31319c4ef7ba0a";
    private static final String HIP_GEOMETRY_SHA256 =
        "4e8eede49aabd4fb71f08dd0b8b87297a9e78ed36f05c3caa4e63de5f469cceb";

    private static final double FIXED_UTC_FROM_TAI_S = -37.0;
    private static final String FRAME_REFERENCE =
        "2025-01-15T00:00:00.000000000";
    private static final double THETA0_RAD = 1.2345678901234567;
    private static final double OMEGA_RAD_S = 7.2921150e-5;

    private static final TimeScale FIXED_UTC =
        new ConstantOffsetTimeScale("FIXED_UTC_TAI_MINUS_37",
                                    new TimeOffset(-37L, 0L));
    private static final AbsoluteDate FRAME_REFERENCE_DATE =
        new AbsoluteDate(FRAME_REFERENCE, FIXED_UTC);
    private static final Frame SYNTHETIC_ECI = Frame.getRoot();
    private static final Frame SYNTHETIC_BODY =
        new Frame(SYNTHETIC_ECI,
                  new SyntheticRotationProvider(FRAME_REFERENCE_DATE),
                  "SYNTHETIC_ROTATING_WGS84_BODY",
                  false);
    private static final OneAxisEllipsoid EARTH =
        new OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                             Constants.WGS84_EARTH_FLATTENING,
                             SYNTHETIC_BODY);

    private static final DriverProfile A =
        new DriverProfile("A", 90.0, 100.0, 95.0, 105.0, 100.0, 110.0,
                          105.0, 115.0, -20.0);
    private static final DriverProfile B =
        new DriverProfile("B", 140.0, 130.0, 150.0, 135.0, 145.0, 132.0,
                          155.0, 138.0, 60.0);
    private static final DriverProfile C =
        new DriverProfile("C", 220.0, 180.0, 205.0, 175.0, 198.0, 170.0,
                          230.0, 185.0, 180.0);
    private static final DriverProfile D =
        new DriverProfile("D", 75.0, 85.0, 80.0, 90.0, 78.0, 88.0,
                          82.0, 92.0, -50.0);

    private static final CaseDef[] CASES = {
        c("case_01", "2025-01-15T23:59:59.500000000", 0.0,    0.0,  200000.0,
          "day_boundary_before", "A_to_B", A, 0),
        c("case_02", "2025-01-16T00:00:00.500000000", 45.0,  30.0,  400000.0,
          "day_boundary_after", "A_to_B", B, 1),
        c("case_03", "2025-01-16T06:00:00.000000000", -45.0, -60.0, 800000.0,
          "interior", "none", B, 2),
        c("case_04", "2025-01-16T12:00:00.000000000", 80.0,   90.0, 1500000.0,
          "interior", "none", B, 3),
        c("case_05", "2025-01-16T18:00:00.000000000", -80.0, -120.0, 200000.0,
          "interior", "none", B, 4),
        c("case_06", "2025-01-16T23:59:59.500000000", 0.0,   150.0, 400000.0,
          "driver_boundary_before", "B_to_C", B, 5),
        c("case_07", "2025-01-17T00:00:00.500000000", 45.0, -150.0, 800000.0,
          "driver_boundary_after", "B_to_C", C, 6),
        c("case_08", "2025-01-17T06:00:00.000000000", -45.0, 120.0, 1500000.0,
          "interior", "none", C, 7),
        c("case_09", "2025-01-17T12:00:00.000000000", 80.0,  -90.0, 200000.0,
          "interior", "none", C, 8),
        c("case_10", "2025-01-17T18:00:00.000000000", -80.0, 60.0, 400000.0,
          "interior", "none", C, 9),
        c("case_11", "2025-01-17T23:59:59.500000000", 0.0,   -30.0, 800000.0,
          "pre_utc_midnight", "C_to_D", C, 10),
        c("case_12", "2025-01-18T00:00:00.500000000", 45.0,  15.0, 1500000.0,
          "post_utc_midnight", "C_to_D", D, 11),
        c("case_13", "2025-01-18T06:00:00.000000000", -45.0, 75.0, 200000.0,
          "interior", "none", D, 12),
        c("case_14", "2025-01-18T12:00:00.000000000", 80.0, -135.0, 400000.0,
          "interior", "none", D, 13),
        c("case_15", "2025-01-18T18:00:00.000000000", -80.0, 165.0, 800000.0,
          "interior", "none", D, 14)
    };

    private OrekitJbEciAdapterVectors() {
    }

    public static void main(final String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        if (args.length != 4) {
            throw new IllegalArgumentException(
                "usage: OrekitJbEciAdapterVectors SOURCE OREKIT CORE GEOMETRY");
        }
        final String sourceSha = sha256(args[0]);
        requireSha(args[1], OREKIT_SHA256);
        requireSha(args[2], HIP_CORE_SHA256);
        requireSha(args[3], HIP_GEOMETRY_SHA256);
        final String payload = buildPayload(sourceSha, null);
        final byte[] payloadBytes = payload.getBytes(StandardCharsets.UTF_8);
        final MessageDigest digest = MessageDigest.getInstance("SHA-256");
        digest.update(HASH_DOMAIN.getBytes(StandardCharsets.US_ASCII));
        digest.update((byte) 0);
        digest.update(ByteBuffer.allocate(8).putLong(payloadBytes.length).array());
        final String semanticSha =
            hex(digest.digest(payloadBytes));
        System.out.print(buildPayload(sourceSha, semanticSha));
        System.out.print('\n');
    }

    private static String buildPayload(final String sourceSha,
                                       final String semanticSha) {
        final StringBuilder out = new StringBuilder(100_000);
        out.append('{');
        field(out, "schema", SCHEMA).append(',');
        field(out, "authority", AUTHORITY).append(',');
        if (semanticSha != null) {
            field(out, "semantic_sha256", semanticSha).append(',');
        }
        field(out, "claim_scope", CLAIM).append(',');
        quote(out, "exclusions").append(':');
        stringArray(out, "not GCRF/ITRF validation", "not physical validation",
                    "not production Rust adapter conformance");
        out.append(',');
        appendAuthority(out, sourceSha);
        out.append(',');
        appendCanonicalization(out);
        out.append(',');
        appendTimeAndFrameLaw(out);
        out.append(',');
        appendEarth(out);
        out.append(',');
        appendUnits(out);
        out.append(',');
        quote(out, "cases").append(':').append('[');
        for (int i = 0; i < CASES.length; ++i) {
            if (i > 0) {
                out.append(',');
            }
            appendCase(out, CASES[i]);
        }
        out.append(']').append('}');
        return out.toString();
    }

    private static void appendAuthority(final StringBuilder out,
                                        final String sourceSha) {
        quote(out, "provenance").append(':').append('{');
        field(out, "generator_source_sha256", sourceSha).append(',');
        field(out, "orekit_version", "13.1.2").append(',');
        field(out, "orekit_jar_sha256", OREKIT_SHA256).append(',');
        field(out, "hipparchus_core_version", "4.0.2").append(',');
        field(out, "hipparchus_core_jar_sha256", HIP_CORE_SHA256).append(',');
        field(out, "hipparchus_geometry_version", "4.0.2").append(',');
        field(out, "hipparchus_geometry_jar_sha256", HIP_GEOMETRY_SHA256).append(',');
        field(out, "java_vendor", System.getProperty("java.vendor")).append(',');
        field(out, "java_version", System.getProperty("java.version")).append(',');
        field(out, "java_vm_name", System.getProperty("java.vm.name")).append(',');
        field(out, "java_vm_version", System.getProperty("java.vm.version")).append(',');
        field(out, "java_specification_version",
              System.getProperty("java.specification.version")).append(',');
        field(out, "os_arch", System.getProperty("os.arch")).append(',');
        field(out, "file_encoding", System.getProperty("file.encoding")).append(',');
        field(out, "compile_flags", "javac --release 8 -encoding UTF-8").append(',');
        field(out, "runtime_flags",
              "java -Xint -Dfile.encoding=UTF-8 -Duser.language=en " +
              "-Duser.country=US -Duser.timezone=UTC " +
              "-Dorekit.data.path=<invalid> -Djava.io.tmpdir=<temp> " +
              "-Duser.home=<temp> -Djava.util.prefs.userRoot=<temp> " +
              "-XX:-UsePerfData -XX:+UseSerialGC -Xms32m -Xmx256m " +
              "-cp <sealed classpath>");
        out.append('}');
    }

    private static void appendCanonicalization(final StringBuilder out) {
        quote(out, "canonicalization").append(':').append('{');
        field(out, "encoding", "UTF-8").append(',');
        field(out, "json", "RFC8259 minified, source-declared key order, LF terminator").append(',');
        field(out, "f64",
              "lowercase 16-digit 0x-prefixed raw IEEE-754 binary64 string; signed zero preserved").append(',');
        field(out, "semantic_hash_domain", HASH_DOMAIN).append(',');
        field(out, "semantic_hash_algorithm", HASH_ALGORITHM);
        out.append('}');
    }

    private static void appendTimeAndFrameLaw(final StringBuilder out) {
        quote(out, "time_and_frame_law").append(':').append('{');
        field(out, "time_scale", "FIXED_UTC_TAI_MINUS_37").append(',');
        f64Field(out, "fixed_utc_offset_from_tai_s", FIXED_UTC_FROM_TAI_S).append(',');
        field(out, "leap_second_policy", "none; fixed offset over declared corpus").append(',');
        field(out, "eci_frame",
              "Orekit Frame.getRoot identity used as synthetic ECI only").append(',');
        field(out, "body_frame", "SYNTHETIC_ROTATING_WGS84_BODY").append(',');
        field(out, "transform_scope", "position transform; no velocity authority").append(',');
        field(out, "rotation_convention",
              "parent ECI to child body VECTOR_OPERATOR rotation about +Z").append(',');
        field(out, "angle_law",
              "theta=theta0+omega*durationFrom(reference_epoch)").append(',');
        field(out, "reference_epoch_fixed_utc", FRAME_REFERENCE).append(',');
        f64Field(out, "theta0_rad", THETA0_RAD).append(',');
        f64Field(out, "omega_rad_s", OMEGA_RAD_S).append(',');
        quote(out, "external_data").append(':');
        stringArray(out);
        out.append('}');
    }

    private static void appendEarth(final StringBuilder out) {
        quote(out, "earth").append(':').append('{');
        field(out, "shape", "WGS84 OneAxisEllipsoid").append(',');
        f64Field(out, "equatorial_radius_m",
                 Constants.WGS84_EARTH_EQUATORIAL_RADIUS).append(',');
        f64Field(out, "flattening", Constants.WGS84_EARTH_FLATTENING);
        out.append('}');
    }

    private static void appendUnits(final StringBuilder out) {
        quote(out, "units").append(':').append('{');
        field(out, "cartesian", "m").append(',');
        field(out, "angle", "rad").append(',');
        field(out, "altitude", "m").append(',');
        field(out, "density", "kg/m^3").append(',');
        field(out, "mjd", "fixed-offset UTC days").append(',');
        field(out, "f10", "1e-22 W/(m^2 Hz)").append(',');
        field(out, "s10_xm10_y10", "JB2008 scaled index units").append(',');
        field(out, "dstdtc", "K");
        out.append('}');
    }

    private static void appendCase(final StringBuilder out, final CaseDef def) {
        final AbsoluteDate date = new AbsoluteDate(def.epoch, FIXED_UTC);
        final double latitude = FastMath.toRadians(def.latitudeDeg);
        final double longitude = FastMath.toRadians(def.longitudeDeg);
        final GeodeticPoint design =
            new GeodeticPoint(latitude, longitude, def.altitudeM);
        final Vector3D satelliteBody = EARTH.transform(design);
        final Vector3D satelliteEci =
            SYNTHETIC_BODY.getStaticTransformTo(SYNTHETIC_ECI, date)
                          .transformPosition(satelliteBody);
        final FixedSun sun = new FixedSun(def.sunEci, SYNTHETIC_ECI);
        final InspectableJB2008 model =
            new InspectableJB2008(def.drivers, sun, EARTH, FIXED_UTC);

        final Transform eciToBody =
            SYNTHETIC_ECI.getTransformTo(SYNTHETIC_BODY, date);
        final Vector3D satelliteBodyExpected =
            eciToBody.transformPosition(satelliteEci);
        final Vector3D sunBody = sun.getPosition(date, SYNTHETIC_BODY);
        final GeodeticPoint satelliteGeodetic =
            EARTH.transform(satelliteEci, SYNTHETIC_ECI, date);
        final GeodeticPoint sunGeodetic =
            EARTH.transform(sunBody, SYNTHETIC_BODY, date);
        final DateTimeComponents components = date.getComponents(FIXED_UTC);
        final double mjd =
            components.getDate().getMJD() +
            components.getTime().getSecondsInLocalDay() / Constants.JULIAN_DAY;
        final double density =
            model.getDensity(date, satelliteEci, SYNTHETIC_ECI);
        final double replay =
            model.replay(date,
                         sunGeodetic.getLongitude(),
                         sunGeodetic.getLatitude(),
                         satelliteGeodetic.getLongitude(),
                         satelliteGeodetic.getLatitude(),
                         satelliteGeodetic.getAltitude());
        if (Double.doubleToRawLongBits(density) !=
            Double.doubleToRawLongBits(replay)) {
            throw new IllegalStateException(def.id + " primitive replay mismatch");
        }

        out.append('{');
        field(out, "id", def.id).append(',');
        quote(out, "epoch").append(':').append('{');
        field(out, "fixed_utc", def.epoch).append(',');
        f64Field(out, "tai_minus_fixed_utc_s", -FIXED_UTC_FROM_TAI_S).append(',');
        f64Field(out, "seconds_from_frame_reference",
                 date.durationFrom(FRAME_REFERENCE_DATE));
        out.append('}').append(',');
        quote(out, "boundary").append(':').append('{');
        field(out, "tag", def.boundaryTag).append(',');
        field(out, "driver_transition", def.driverTransition).append(',');
        field(out, "driver_profile_id", def.drivers.id);
        out.append('}').append(',');
        quote(out, "design").append(':').append('{');
        f64Field(out, "geodetic_latitude_rad", latitude).append(',');
        f64Field(out, "geodetic_longitude_rad", longitude).append(',');
        f64Field(out, "altitude_m", def.altitudeM);
        out.append('}').append(',');
        quote(out, "inputs").append(':').append('{');
        vectorField(out, "satellite_eci_m", satelliteEci).append(',');
        vectorField(out, "earth_to_sun_eci_m", def.sunEci).append(',');
        appendDrivers(out, def.drivers);
        out.append('}').append(',');
        quote(out, "expected").append(':').append('{');
        matrixField(out, "eci_to_body_matrix",
                    eciToBody.getRotation().getMatrix()).append(',');
        vectorField(out, "satellite_body_m", satelliteBodyExpected).append(',');
        vectorField(out, "sun_body_m", sunBody).append(',');
        geodeticField(out, "satellite_geodetic", satelliteGeodetic).append(',');
        geodeticField(out, "sun_geodetic", sunGeodetic).append(',');
        appendPrimitives(out, mjd, sunGeodetic, satelliteGeodetic, def.drivers);
        out.append(',');
        f64Field(out, "density_kg_m3", density);
        out.append('}').append('}');
    }

    private static void appendDrivers(final StringBuilder out,
                                      final DriverProfile d) {
        quote(out, "jb_drivers").append(':').append('{');
        f64Field(out, "f10", d.f10).append(',');
        f64Field(out, "f10b", d.f10b).append(',');
        f64Field(out, "s10", d.s10).append(',');
        f64Field(out, "s10b", d.s10b).append(',');
        f64Field(out, "xm10", d.xm10).append(',');
        f64Field(out, "xm10b", d.xm10b).append(',');
        f64Field(out, "y10", d.y10).append(',');
        f64Field(out, "y10b", d.y10b).append(',');
        f64Field(out, "dstdtc", d.dstdtc);
        out.append('}');
    }

    private static void appendPrimitives(final StringBuilder out,
                                         final double mjd,
                                         final GeodeticPoint sun,
                                         final GeodeticPoint satellite,
                                         final DriverProfile d) {
        quote(out, "jb_primitive_inputs").append(':').append('{');
        f64Field(out, "date_mjd_fixed_utc", mjd).append(',');
        f64Field(out, "sun_longitude_rad_as_sunRA",
                 sun.getLongitude()).append(',');
        f64Field(out, "sun_geodetic_latitude_rad_as_sunDecli",
                 sun.getLatitude()).append(',');
        f64Field(out, "satellite_geodetic_longitude_rad_as_satLon",
                 satellite.getLongitude()).append(',');
        f64Field(out, "satellite_geodetic_latitude_rad_as_satLat",
                 satellite.getLatitude()).append(',');
        f64Field(out, "satellite_ellipsoidal_altitude_m_as_satAlt",
                 satellite.getAltitude()).append(',');
        f64Field(out, "f10", d.f10).append(',');
        f64Field(out, "f10b", d.f10b).append(',');
        f64Field(out, "s10", d.s10).append(',');
        f64Field(out, "s10b", d.s10b).append(',');
        f64Field(out, "xm10", d.xm10).append(',');
        f64Field(out, "xm10b", d.xm10b).append(',');
        f64Field(out, "y10", d.y10).append(',');
        f64Field(out, "y10b", d.y10b).append(',');
        f64Field(out, "dstdtc", d.dstdtc);
        out.append('}');
    }

    private static StringBuilder geodeticField(final StringBuilder out,
                                               final String name,
                                               final GeodeticPoint point) {
        quote(out, name).append(':').append('{');
        f64Field(out, "longitude_rad", point.getLongitude()).append(',');
        f64Field(out, "latitude_rad", point.getLatitude()).append(',');
        f64Field(out, "altitude_m", point.getAltitude());
        return out.append('}');
    }

    private static StringBuilder vectorField(final StringBuilder out,
                                             final String name,
                                             final Vector3D vector) {
        quote(out, name).append(':').append('[');
        f64(out, vector.getX()).append(',');
        f64(out, vector.getY()).append(',');
        f64(out, vector.getZ());
        return out.append(']');
    }

    private static StringBuilder matrixField(final StringBuilder out,
                                             final String name,
                                             final double[][] matrix) {
        quote(out, name).append(':').append('[');
        for (int row = 0; row < 3; ++row) {
            if (row > 0) {
                out.append(',');
            }
            out.append('[');
            for (int column = 0; column < 3; ++column) {
                if (column > 0) {
                    out.append(',');
                }
                f64(out, matrix[row][column]);
            }
            out.append(']');
        }
        return out.append(']');
    }

    private static StringBuilder f64Field(final StringBuilder out,
                                          final String name,
                                          final double value) {
        quote(out, name).append(':');
        return f64(out, value);
    }

    private static StringBuilder f64(final StringBuilder out,
                                     final double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("nonfinite f64");
        }
        return quote(out, "0x" + hex16(Double.doubleToRawLongBits(value)));
    }

    private static StringBuilder field(final StringBuilder out,
                                       final String name,
                                       final String value) {
        quote(out, name).append(':');
        return quote(out, value);
    }

    private static StringBuilder quote(final StringBuilder out,
                                       final String value) {
        out.append('"');
        for (int i = 0; i < value.length(); ++i) {
            final char c = value.charAt(i);
            if (c == '"' || c == '\\') {
                out.append('\\');
            }
            if (c < 0x20 || c > 0x7e) {
                throw new IllegalArgumentException("non-ASCII JSON string");
            }
            out.append(c);
        }
        return out.append('"');
    }

    private static void stringArray(final StringBuilder out,
                                    final String... values) {
        out.append('[');
        for (int i = 0; i < values.length; ++i) {
            if (i > 0) {
                out.append(',');
            }
            quote(out, values[i]);
        }
        out.append(']');
    }

    private static String hex16(final long value) {
        final String raw = Long.toHexString(value);
        final StringBuilder out = new StringBuilder(16);
        for (int i = raw.length(); i < 16; ++i) {
            out.append('0');
        }
        return out.append(raw).toString();
    }

    private static String hex(final byte[] bytes) {
        final StringBuilder out = new StringBuilder(bytes.length * 2);
        for (final byte b : bytes) {
            final int unsigned = b & 0xff;
            if (unsigned < 16) {
                out.append('0');
            }
            out.append(Integer.toHexString(unsigned));
        }
        return out.toString();
    }

    private static String sha256(final String path) throws Exception {
        final MessageDigest digest = MessageDigest.getInstance("SHA-256");
        return hex(digest.digest(Files.readAllBytes(Paths.get(path))));
    }

    private static void requireSha(final String path,
                                   final String expected) throws Exception {
        final String actual = sha256(path);
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(
                "dependency SHA-256 mismatch for " + path +
                ": expected " + expected + ", got " + actual);
        }
    }

    private static CaseDef c(final String id, final String epoch,
                             final double latitudeDeg,
                             final double longitudeDeg,
                             final double altitudeM,
                             final String boundaryTag,
                             final String driverTransition,
                             final DriverProfile drivers,
                             final int sunIndex) {
        final Vector3D sun =
            new Vector3D(130_000_000_000.0 - sunIndex * 20_000_000_000.0,
                         -100_000_000_000.0 + sunIndex * 15_000_000_000.0,
                         -45_000_000_000.0 + sunIndex * 7_000_000_000.0);
        return new CaseDef(id, epoch, latitudeDeg, longitudeDeg, altitudeM,
                           boundaryTag, driverTransition, drivers, sun);
    }

    private static final class CaseDef {
        private final String id;
        private final String epoch;
        private final double latitudeDeg;
        private final double longitudeDeg;
        private final double altitudeM;
        private final String boundaryTag;
        private final String driverTransition;
        private final DriverProfile drivers;
        private final Vector3D sunEci;

        CaseDef(final String id, final String epoch,
                final double latitudeDeg, final double longitudeDeg,
                final double altitudeM, final String boundaryTag,
                final String driverTransition,
                final DriverProfile drivers, final Vector3D sunEci) {
            this.id = id;
            this.epoch = epoch;
            this.latitudeDeg = latitudeDeg;
            this.longitudeDeg = longitudeDeg;
            this.altitudeM = altitudeM;
            this.boundaryTag = boundaryTag;
            this.driverTransition = driverTransition;
            this.drivers = drivers;
            this.sunEci = sunEci;
        }
    }

    private static final class DriverProfile
        implements JB2008InputParameters {
        private static final long serialVersionUID = 1L;
        private final String id;
        private final double f10;
        private final double f10b;
        private final double s10;
        private final double s10b;
        private final double xm10;
        private final double xm10b;
        private final double y10;
        private final double y10b;
        private final double dstdtc;

        DriverProfile(final String id,
                      final double f10, final double f10b,
                      final double s10, final double s10b,
                      final double xm10, final double xm10b,
                      final double y10, final double y10b,
                      final double dstdtc) {
            this.id = id;
            this.f10 = f10;
            this.f10b = f10b;
            this.s10 = s10;
            this.s10b = s10b;
            this.xm10 = xm10;
            this.xm10b = xm10b;
            this.y10 = y10;
            this.y10b = y10b;
            this.dstdtc = dstdtc;
        }

        public AbsoluteDate getMinDate() { return AbsoluteDate.PAST_INFINITY; }
        public AbsoluteDate getMaxDate() { return AbsoluteDate.FUTURE_INFINITY; }
        public double getF10(final AbsoluteDate date) { return f10; }
        public double getF10B(final AbsoluteDate date) { return f10b; }
        public double getS10(final AbsoluteDate date) { return s10; }
        public double getS10B(final AbsoluteDate date) { return s10b; }
        public double getXM10(final AbsoluteDate date) { return xm10; }
        public double getXM10B(final AbsoluteDate date) { return xm10b; }
        public double getY10(final AbsoluteDate date) { return y10; }
        public double getY10B(final AbsoluteDate date) { return y10b; }
        public double getDSTDTC(final AbsoluteDate date) { return dstdtc; }
    }

    private static final class FixedSun implements ExtendedPositionProvider {
        private final Vector3D eciPosition;
        private final Frame source;

        FixedSun(final Vector3D eciPosition, final Frame source) {
            this.eciPosition = eciPosition;
            this.source = source;
        }

        @Override
        public Vector3D getPosition(final AbsoluteDate date,
                                    final Frame frame) {
            return source.getStaticTransformTo(frame, date)
                         .transformPosition(eciPosition);
        }

        @Override
        public <T extends CalculusFieldElement<T>>
            FieldVector3D<T> getPosition(final FieldAbsoluteDate<T> date,
                                         final Frame frame) {
            return source.getStaticTransformTo(frame, date)
                         .transformPosition(eciPosition);
        }
    }

    private static final class SyntheticRotationProvider
        implements TransformProvider {
        private final AbsoluteDate reference;

        SyntheticRotationProvider(final AbsoluteDate reference) {
            this.reference = reference;
        }

        private Rotation rotation(final AbsoluteDate date) {
            final double angle =
                THETA0_RAD + OMEGA_RAD_S * date.durationFrom(reference);
            return new Rotation(Vector3D.PLUS_K, angle,
                                RotationConvention.VECTOR_OPERATOR);
        }

        private <T extends CalculusFieldElement<T>>
            FieldRotation<T> rotation(final FieldAbsoluteDate<T> date) {
            final T angle = date.durationFrom(reference)
                                .multiply(OMEGA_RAD_S)
                                .add(THETA0_RAD);
            return new FieldRotation<>(
                new FieldVector3D<>(date.getField(), Vector3D.PLUS_K),
                angle, RotationConvention.VECTOR_OPERATOR);
        }

        @Override
        public Transform getTransform(final AbsoluteDate date) {
            return new Transform(date, rotation(date),
                                 new Vector3D(OMEGA_RAD_S, Vector3D.PLUS_K));
        }

        @Override
        public <T extends CalculusFieldElement<T>>
            FieldTransform<T> getTransform(final FieldAbsoluteDate<T> date) {
            return new FieldTransform<>(
                date, rotation(date),
                new FieldVector3D<>(date.getField(),
                    new Vector3D(OMEGA_RAD_S, Vector3D.PLUS_K)));
        }
    }

    private static final class InspectableJB2008 extends JB2008 {
        InspectableJB2008(final JB2008InputParameters parameters,
                          final ExtendedPositionProvider sun,
                          final OneAxisEllipsoid earth,
                          final TimeScale utc) {
            super(parameters, sun, earth, utc);
        }

        double replay(final AbsoluteDate date,
                      final double sunLongitude,
                      final double sunLatitude,
                      final double satelliteLongitude,
                      final double satelliteLatitude,
                      final double satelliteAltitude) {
            return computeDensity(date, sunLongitude, sunLatitude,
                                  satelliteLongitude, satelliteLatitude,
                                  satelliteAltitude);
        }
    }
}
