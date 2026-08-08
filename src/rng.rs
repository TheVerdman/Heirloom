use crate::shape::checked_numel;
use crate::{Result, Tensor, TensorError};

#[derive(Clone, Debug)]
pub struct HeirloomRng {
    state: u64,
    cached_normal: Option<f32>,
}

impl HeirloomRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    pub fn from_state(state: u64) -> Self {
        Self {
            state,
            cached_normal: None,
        }
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn uniform_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }

    pub fn uniform_range(&mut self, low: f32, high: f32) -> Result<f32> {
        if low >= high {
            return Err(TensorError::InvalidOperation(format!(
                "uniform_range requires low < high, got low={low}, high={high}"
            )));
        }
        Ok(low + (high - low) * self.uniform_f32())
    }

    pub fn normal_f32(&mut self) -> f32 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }

        let u1 = self.uniform_f32().max(f32::MIN_POSITIVE);
        let u2 = self.uniform_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }

    pub fn uniform_tensor(
        &mut self,
        shape: &[usize],
        low: f32,
        high: f32,
        requires_grad: bool,
    ) -> Result<Tensor> {
        let len = checked_numel(shape)?;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(self.uniform_range(low, high)?);
        }
        Tensor::from_vec(data, shape, requires_grad)
    }

    pub fn normal_tensor(
        &mut self,
        shape: &[usize],
        mean: f32,
        stddev: f32,
        requires_grad: bool,
    ) -> Result<Tensor> {
        if stddev < 0.0 {
            return Err(TensorError::InvalidOperation(format!(
                "normal_tensor requires non-negative stddev, got {stddev}"
            )));
        }
        let len = checked_numel(shape)?;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(mean + stddev * self.normal_f32());
        }
        Tensor::from_vec(data, shape, requires_grad)
    }
}
