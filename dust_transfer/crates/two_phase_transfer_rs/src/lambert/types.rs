use nalgebra::Vector3;

/// Eigen-compatible Vector3d alias
pub type Vec3 = Vector3<f64>;

/// Zero-copy view from raw array (same memory layout)
#[inline]
#[must_use]
pub const fn vec3_from_slice(s: &[f64; 3]) -> Vec3 {
    Vec3::new(s[0], s[1], s[2])
}

/// Zero-copy conversion to array
#[inline]
#[must_use]
pub fn vec3_to_array(v: &Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}
