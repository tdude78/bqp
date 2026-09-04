use jb_rs::drivers::{compiled_drivers, compiled_identity, validate_utc_arc, UtcJulianDay};
use std::sync::Arc;
use std::thread;

#[test]
fn compiled_provider_reuses_one_immutable_arc_and_reports_identity() {
    let first = compiled_drivers().expect("compiled SET drivers");
    let second = compiled_drivers().expect("compiled SET drivers");
    assert!(Arc::ptr_eq(&first, &second));

    let identity = compiled_identity().expect("compiled identity");
    assert_eq!(identity.kernel_name, "orekit_13_1_2_jb2008_f64_kernel");
    assert_eq!(identity.kernel_version, "v1");
    assert_eq!(
        identity.manifest_sha256,
        "09d2cb0cdb5cb805f6b10e9e2141e02d9320c91071491e2b32da87857deac01c"
    );
    assert_eq!(
        identity.solfsmy_sha256,
        "2e8f31bd3294b1982b23bb475b560f00593c90ec0c3e50667ce4e068794efe80"
    );
    assert_eq!(
        identity.dtcfile_sha256,
        "e0d44fc1dbb176693159ebd98ee481319a207c623d7b954509bc404d4aac7594"
    );
    assert_eq!(
        identity.license_sha256,
        "9d2ec826044266557880c7863443ac2609312db39780b1c8239f5578dd75d387"
    );
    assert!(identity
        .set_release
        .starts_with("F10, S10, M10, Y10 data release 8_1_0"));
    assert_eq!(
        identity.solfsmy_coverage_start_jd.to_bits(),
        2_450_450.0_f64.to_bits()
    );
    assert_eq!(
        identity.solfsmy_coverage_end_jd.to_bits(),
        2_461_195.0_f64.to_bits()
    );
    assert_eq!(
        identity.dtc_coverage_start_jd.to_bits(),
        2_450_450.0_f64.to_bits()
    );
    assert_eq!(
        identity.dtc_coverage_end_jd.to_bits(),
        2_461_195.0_f64.to_bits()
    );
}

#[test]
fn typed_utc_lookup_and_arc_validation_enforce_lag_and_next_day_dtc() {
    let utc_jd = UtcJulianDay::new(2_459_528.0).expect("finite UTC JD");
    let converted_mjd = utc_jd.to_utc_mjd().expect("UTC MJD conversion");
    assert_eq!(
        converted_mjd
            .to_utc_jd()
            .expect("round trip")
            .as_f64()
            .to_bits(),
        utc_jd.as_f64().to_bits()
    );

    let drivers = compiled_drivers().expect("compiled SET drivers");
    assert_eq!(
        drivers
            .lookup_utc_mjd(converted_mjd)
            .expect("typed MJD lookup"),
        drivers.lookup_utc_jd(utc_jd).expect("typed JD lookup")
    );
    validate_utc_arc(utc_jd, utc_jd).expect("covered UTC arc");

    let missing_y10 = UtcJulianDay::new(2_450_454.0).expect("finite UTC JD");
    assert!(validate_utc_arc(missing_y10, missing_y10).is_err());

    let missing_next_dtc = UtcJulianDay::new(2_461_195.0).expect("finite UTC JD");
    assert!(validate_utc_arc(missing_next_dtc, missing_next_dtc).is_err());

    assert!(validate_utc_arc(missing_next_dtc, utc_jd).is_err());
}

#[test]
fn compiled_provider_arc_identity_is_shared_across_threads() {
    let expected = compiled_drivers().expect("compiled SET drivers");
    let workers: Vec<_> = (0..8)
        .map(|_| thread::spawn(|| compiled_drivers().expect("compiled SET drivers")))
        .collect();

    for worker in workers {
        let actual = worker.join().expect("worker completion");
        assert!(Arc::ptr_eq(&expected, &actual));
    }
}
