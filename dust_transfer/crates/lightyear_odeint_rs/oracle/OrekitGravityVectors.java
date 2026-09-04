import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.ICGEMFormatReader;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.forces.gravity.potential.RawSphericalHarmonicsProvider;
import org.orekit.forces.gravity.potential.TideSystem;
import org.orekit.frames.Frame;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.propagation.SpacecraftState;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;

/** DIR-R6 d/o5 source-output generator; its output is not an accepted fixture. */
public strictfp final class OrekitGravityVectors {
    private static final String SCHEMA = "part_a_orekit_dir_r6_gravity_v1";
    private static final String DOMAIN = "PART_A_OREKIT_DIR_R6_GRAVITY_V1";
    private static final String AUTHORITY =
        "Official ICGEM/GFZ GO_CONS_GCF_2_DIR_R6 with Orekit 13.1.2 d/o5 comparator";
    private static final String CLAIM =
        "Orekit d/o5 noncentral and full body-fixed gravity comparator; no frame transform, propagation, or Task 5 claim";
    private static final String OREKIT_SHA256 =
        "89c2060c60dbe194a87dddcf3bb8343ebd16733958efe4dcc996cebbbeed655d";
    private static final String CORE_SHA256 =
        "7c56992f3af64429d871c33c00808ee5db5d9ed56b395b5d3d31319c4ef7ba0a";
    private static final String GEOMETRY_SHA256 =
        "4e8eede49aabd4fb71f08dd0b8b87297a9e78ed36f05c3caa4e63de5f469cceb";
    private static final String JAR_AGGREGATE_SHA256 =
        "7e3b504bfd38b0d6713b959085e7fcfba8a6ae635bf4b769006d816d6b7e7d24";
    private static final String DIR_R6_SHA256 =
        "4da4a476418553c2243c0dbc79515bb3a419f3175dea3e38c58843cb14fcff7b";
    private static final String DERIVED_SHA256 =
        "983f035818399f9cb27f1e8c604cb62b3e72d650aa4cbfadb31b1e7c4fe61f09";
    private static final double MU = 3.986004415e14;
    private static final double AE = 6378136.46;
    private static final double ABS_TOL = 5.0e-11;
    private static final double REL_TOL = 2.0e-12;

    private static final String[] IDS = {
        "equatorial_leo", "generic_leo", "near_polar_leo", "generic_meo", "generic_geo"
    };
    private static final double[][] POSITIONS_M = {
        {6778136.46, 0.0, 0.0},
        {4000000.0, -5000000.0, 3000000.0},
        {100000.0, -200000.0, 7000000.0},
        {-12000000.0, 16000000.0, -9000000.0},
        {42164000.0, -3000000.0, 500000.0}
    };

    private OrekitGravityVectors() { }

    public static void main(final String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        if (args.length != 6) {
            throw new IllegalArgumentException(
                "usage: OrekitGravityVectors DIR_R6_GFC GENERATOR_SOURCE DERIVED_D15 OREKIT_JAR HIPPARCHUS_CORE_JAR HIPPARCHUS_GEOMETRY_JAR");
        }
        requireSha(args[0], DIR_R6_SHA256);
        requireSha(args[2], DERIVED_SHA256);
        requireSha(args[3], OREKIT_SHA256);
        requireSha(args[4], CORE_SHA256);
        requireSha(args[5], GEOMETRY_SHA256);
        requireJarAggregate(args[3], args[4], args[5]);

        final String generatorSha = sha256(args[1]);
        final ICGEMFormatReader reader = new ICGEMFormatReader("GO_CONS_GCF_2_DIR_R6\\.gfc", false);
        try (FileInputStream input = new FileInputStream(args[0])) {
            reader.loadData(input, "GO_CONS_GCF_2_DIR_R6.gfc");
        }
        final RawSphericalHarmonicsProvider raw = reader.getProvider(true, 5, 5);
        requireBits(raw.getMu(), MU, "GM");
        requireBits(raw.getAe(), AE, "reference radius");
        if (raw.getTideSystem() != TideSystem.TIDE_FREE) {
            throw new IllegalStateException("DIR-R6 tide system must be tide_free");
        }
        final double[][] c = new double[6][];
        final double[][] s = new double[6][];
        final RawSphericalHarmonicsProvider.RawSphericalHarmonics h =
            raw.onDate(AbsoluteDate.J2000_EPOCH);
        for (int n = 0; n <= 5; ++n) {
            c[n] = new double[n + 1];
            s[n] = new double[n + 1];
            for (int m = 0; m <= n; ++m) {
                c[n][m] = h.getRawCnm(n, m);
                s[n][m] = h.getRawSnm(n, m);
            }
        }
        final NormalizedSphericalHarmonicsProvider provider =
            GravityFieldFactory.getNormalizedProvider(AE, MU, TideSystem.TIDE_FREE, c, s);
        final Frame body = Frame.getRoot();
        final HolmesFeatherstoneAttractionModel model =
            new HolmesFeatherstoneAttractionModel(body, provider);
        final CaseResult[] cases = evaluate(model, body);
        final String semanticSha = semanticSha(generatorSha, cases);
        System.out.print(payload(generatorSha, semanticSha, cases));
        System.out.print('\n');
    }

    private static CaseResult[] evaluate(final HolmesFeatherstoneAttractionModel model,
                                         final Frame body) {
        final CaseResult[] results = new CaseResult[IDS.length];
        for (int i = 0; i < IDS.length; ++i) {
            final Vector3D position = new Vector3D(POSITIONS_M[i][0], POSITIONS_M[i][1], POSITIONS_M[i][2]);
            final SpacecraftState state = new SpacecraftState(new CartesianOrbit(
                new PVCoordinates(position, Vector3D.ZERO), body, AbsoluteDate.J2000_EPOCH, MU));
            final Vector3D noncentral = model.acceleration(state, new double[] {MU});
            final Vector3D pointMass = position.scalarMultiply(-MU / Math.pow(position.getNorm(), 3));
            results[i] = new CaseResult(IDS[i], position, noncentral, pointMass, noncentral.add(pointMass));
        }
        return results;
    }

    private static String payload(final String generatorSha, final String semanticSha,
                                  final CaseResult[] cases) {
        final StringBuilder out = new StringBuilder(8192);
        out.append('{');
        stringField(out, "schema", SCHEMA).append(',');
        stringField(out, "authority", AUTHORITY).append(',');
        stringField(out, "claim_scope", CLAIM).append(',');
        stringField(out, "semantic_sha256", semanticSha).append(',');
        out.append("\"provenance\":{");
        stringField(out, "generator_source_sha256", generatorSha).append(',');
        stringField(out, "source_gfc_sha256", DIR_R6_SHA256).append(',');
        stringField(out, "derived_d15_sha256", DERIVED_SHA256).append(',');
        stringField(out, "jar_aggregate_sha256", JAR_AGGREGATE_SHA256).append(',');
        stringField(out, "orekit_version", "13.1.2").append(',');
        stringField(out, "hipparchus_core_version", "4.0.2").append(',');
        stringField(out, "hipparchus_geometry_version", "4.0.2").append(',');
        stringField(out, "orekit_jar_sha256", OREKIT_SHA256).append(',');
        stringField(out, "hipparchus_core_jar_sha256", CORE_SHA256).append(',');
        stringField(out, "hipparchus_geometry_jar_sha256", GEOMETRY_SHA256);
        out.append("},\"model\":{");
        stringField(out, "name", "GO_CONS_EGM_GOC_2__20091009T000000_20131020T235959_0201").append(',');
        stringField(out, "tide_system", "tide_free").append(',');
        stringField(out, "normalization", "fully_normalized").append(',');
        hexField(out, "gm_m3_s2", MU).append(',');
        hexField(out, "reference_radius_m", AE).append(',');
        hexField(out, "stored_degree", 15.0).append(',');
        hexField(out, "stored_order", 15.0).append(',');
        hexField(out, "runtime_degree", 5.0).append(',');
        hexField(out, "runtime_order", 5.0);
        out.append("},\"evaluation\":{");
        stringField(out, "body_frame", "Frame.getRoot identity").append(',');
        stringField(out, "epoch", "J2000_EPOCH").append(',');
        stringField(out, "units", "m,m/s^2").append(',');
        hexField(out, "absolute_tolerance_m_s2", ABS_TOL).append(',');
        hexField(out, "relative_tolerance", REL_TOL);
        out.append("},\"cases\":[");
        for (int i = 0; i < cases.length; ++i) {
            if (i != 0) { out.append(','); }
            casePayload(out, cases[i]);
        }
        out.append("]}");
        return out.toString();
    }

    private static void casePayload(final StringBuilder out, final CaseResult result) {
        out.append('{');
        stringField(out, "name", result.name).append(',');
        vectorField(out, "position_m", result.position).append(',');
        vectorField(out, "orekit_noncentral_m_s2", result.noncentral).append(',');
        vectorField(out, "point_mass_m_s2", result.pointMass).append(',');
        vectorField(out, "full_m_s2", result.full);
        out.append('}');
    }

    private static String semanticSha(final String generatorSha, final CaseResult[] cases)
        throws Exception {
        final MessageDigest digest = MessageDigest.getInstance("SHA-256");
        digest.update(DOMAIN.getBytes(StandardCharsets.US_ASCII));
        digest.update((byte) 0);
        scalar(digest, "schema", SCHEMA);
        scalar(digest, "authority", AUTHORITY);
        scalar(digest, "claim_scope", CLAIM);
        scalar(digest, "provenance.generator_source_sha256", generatorSha);
        scalar(digest, "provenance.source_gfc_sha256", DIR_R6_SHA256);
        scalar(digest, "provenance.derived_d15_sha256", DERIVED_SHA256);
        scalar(digest, "provenance.jar_aggregate_sha256", JAR_AGGREGATE_SHA256);
        scalar(digest, "provenance.orekit_version", "13.1.2");
        scalar(digest, "provenance.hipparchus_core_version", "4.0.2");
        scalar(digest, "provenance.hipparchus_geometry_version", "4.0.2");
        scalar(digest, "provenance.orekit_jar_sha256", OREKIT_SHA256);
        scalar(digest, "provenance.hipparchus_core_jar_sha256", CORE_SHA256);
        scalar(digest, "provenance.hipparchus_geometry_jar_sha256", GEOMETRY_SHA256);
        scalar(digest, "model.name", "GO_CONS_EGM_GOC_2__20091009T000000_20131020T235959_0201");
        scalar(digest, "model.tide_system", "tide_free");
        scalar(digest, "model.normalization", "fully_normalized");
        scalar(digest, "model.gm_m3_s2", hex(MU));
        scalar(digest, "model.reference_radius_m", hex(AE));
        scalar(digest, "model.stored_degree", hex(15.0));
        scalar(digest, "model.stored_order", hex(15.0));
        scalar(digest, "model.runtime_degree", hex(5.0));
        scalar(digest, "model.runtime_order", hex(5.0));
        scalar(digest, "evaluation.body_frame", "Frame.getRoot identity");
        scalar(digest, "evaluation.epoch", "J2000_EPOCH");
        scalar(digest, "evaluation.units", "m,m/s^2");
        scalar(digest, "evaluation.absolute_tolerance_m_s2", hex(ABS_TOL));
        scalar(digest, "evaluation.relative_tolerance", hex(REL_TOL));
        for (int i = 0; i < cases.length; ++i) { semanticCase(digest, i, cases[i]); }
        return hex(digest.digest());
    }

    private static void semanticCase(final MessageDigest digest, final int index,
                                     final CaseResult result) {
        final String prefix = "cases[" + index + "].";
        scalar(digest, prefix + "name", result.name);
        scalarVector(digest, prefix + "position_m", result.position);
        scalarVector(digest, prefix + "orekit_noncentral_m_s2", result.noncentral);
        scalarVector(digest, prefix + "point_mass_m_s2", result.pointMass);
        scalarVector(digest, prefix + "full_m_s2", result.full);
    }

    private static void scalarVector(final MessageDigest digest, final String tag,
                                     final Vector3D value) {
        scalar(digest, tag + "[0]", hex(value.getX()));
        scalar(digest, tag + "[1]", hex(value.getY()));
        scalar(digest, tag + "[2]", hex(value.getZ()));
    }

    private static void scalar(final MessageDigest digest, final String tag, final String value) {
        digest.update(tag.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        digest.update(value.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
    }

    private static void requireJarAggregate(final String orekit, final String core,
                                            final String geometry) throws Exception {
        final MessageDigest digest = MessageDigest.getInstance("SHA-256");
        aggregateFile(digest, "hipparchus-core-4.0.2.jar", core);
        aggregateFile(digest, "hipparchus-geometry-4.0.2.jar", geometry);
        aggregateFile(digest, "orekit-13.1.2.jar", orekit);
        if (!JAR_AGGREGATE_SHA256.equals(hex(digest.digest()))) {
            throw new IllegalStateException("JAR aggregate SHA-256 mismatch");
        }
    }

    private static void aggregateFile(final MessageDigest digest, final String name,
                                      final String path) throws Exception {
        final byte[] contents = Files.readAllBytes(Paths.get(path));
        digest.update(name.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        digest.update(Long.toString(contents.length).getBytes(StandardCharsets.US_ASCII));
        digest.update((byte) 0);
        digest.update(contents);
    }

    private static void requireSha(final String path, final String expected) throws Exception {
        if (!expected.equals(sha256(path))) { throw new IllegalStateException("SHA-256 mismatch: " + path); }
    }

    private static String sha256(final String path) throws Exception {
        return hex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Paths.get(path))));
    }

    private static void requireBits(final double actual, final double expected, final String label) {
        if (Double.doubleToRawLongBits(actual) != Double.doubleToRawLongBits(expected)) {
            throw new IllegalStateException(label + " mismatch");
        }
    }

    private static StringBuilder stringField(final StringBuilder out, final String key, final String value) {
        return out.append('"').append(key).append("\":\"").append(value).append('"');
    }

    private static StringBuilder hexField(final StringBuilder out, final String key, final double value) {
        return stringField(out, key, hex(value));
    }

    private static StringBuilder vectorField(final StringBuilder out, final String key, final Vector3D value) {
        out.append('"').append(key).append("\":[");
        stringValue(out, hex(value.getX())).append(',');
        stringValue(out, hex(value.getY())).append(',');
        stringValue(out, hex(value.getZ())).append(']');
        return out;
    }

    private static StringBuilder stringValue(final StringBuilder out, final String value) {
        return out.append('"').append(value).append('"');
    }

    private static String hex(final double value) {
        return String.format(Locale.ROOT, "0x%016x", Double.doubleToRawLongBits(value));
    }

    private static String hex(final byte[] bytes) {
        final StringBuilder out = new StringBuilder(bytes.length * 2);
        for (final byte value : bytes) { out.append(String.format(Locale.ROOT, "%02x", value & 0xff)); }
        return out.toString();
    }

    private static final class CaseResult {
        private final String name;
        private final Vector3D position;
        private final Vector3D noncentral;
        private final Vector3D pointMass;
        private final Vector3D full;

        CaseResult(final String name, final Vector3D position, final Vector3D noncentral,
                   final Vector3D pointMass, final Vector3D full) {
            this.name = name;
            this.position = position;
            this.noncentral = noncentral;
            this.pointMass = pointMass;
            this.full = full;
        }
    }
}
