import java.io.File;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.models.earth.atmosphere.JB2008;
import org.orekit.time.TimeScalesFactory;

public final class OrekitJb2008Vectors {
    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: OrekitJbVectors OREKIT_DATA_DIR");
        }
        DataContext.getDefault().getDataProvidersManager()
            .addProvider(new DirectoryCrawler(new File(args[0])));
        JB2008 model = new JB2008(null, null, null, TimeScalesFactory.getUTC());
        double[] altitudeKm = {
            91, 120, 200, 240, 300, 600, 800, 1000, 1500, 2300, 2500, 3000, 35000
        };
        for (double altitude : altitudeKm) {
            print(model, 52951.003805740744, altitude);
        }
        print(model, 35000.25, 400.0);
    }

    private static void print(JB2008 model, double mjdUtc, double altitudeKm) {
            double rho = model.getDensity(
                mjdUtc,
                3.046653643566772,
                -0.285987757544287,
                1.28211886851503,
                -1.4877186543999,
                altitudeKm * 1000.0,
                91.00,
                137.10,
                108.80,
                123.80,
                116.70,
                128.50,
                168.00,
                138.60,
                43.0
            );
            System.out.printf("%.8f %.0f %.17e 0x%016x%n",
                mjdUtc, altitudeKm, rho, Double.doubleToLongBits(rho));
    }
}
