use jb_rs::jb2008;

#[test]
fn kernel_identity_is_explicit() {
    assert_eq!(
        jb2008::JB2008_MODEL_NAME,
        "orekit_13_1_2_jb2008_f64_kernel_v1"
    );
}
