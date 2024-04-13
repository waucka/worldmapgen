use simdnoise::*;
use simdeez::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use simdeez::avx2::*;
use nalgebra::base::Vector2;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Serialize, Deserialize};

use std::error::Error;
use std::f32::consts::PI;

simd_runtime_generate!(
    pub fn add_vectors(accum: &mut [f32], src: &[f32]) {
        let accum_len = accum.len();
        if accum_len != src.len() {
            panic!("Vector lengths don't match: {} vs. {}", accum_len, src.len());
        }
        let mut i = 0;
        while (i + S::VF32_WIDTH) < accum_len {
            let accum_chunk = S::loadu_ps(&accum[i]);
            let src_chunk = S::loadu_ps(&src[i]);
            let sum_chunk = accum_chunk + src_chunk;
            S::storeu_ps(&mut accum[i], sum_chunk);
            i += S::VF32_WIDTH;
        }
        while i < accum_len {
            accum[i] += src[i];
            i += 1;
        }
    }
);

simd_runtime_generate!(
    pub fn subtract_vectors(accum: &mut [f32], src: &[f32]) {
        let accum_len = accum.len();
        if accum_len != src.len() {
            panic!("Vector lengths don't match: {} vs. {}", accum_len, src.len());
        }
        let mut i = 0;
        while (i + S::VF32_WIDTH) < accum_len {
            let accum_chunk = S::loadu_ps(&accum[i]);
            let src_chunk = S::loadu_ps(&src[i]);
            let sum_chunk = accum_chunk - src_chunk;
            S::storeu_ps(&mut accum[i], sum_chunk);
            i += S::VF32_WIDTH;
        }
        while i < accum_len {
            accum[i] -= src[i];
            i += 1;
        }
    }
);

simd_runtime_generate!(
    pub fn multiply_vectors(accum: &mut [f32], src: &[f32]) {
        let accum_len = accum.len();
        if accum_len != src.len() {
            panic!("Vector lengths don't match: {} vs. {}", accum_len, src.len());
        }
        let mut i = 0;
        while (i + S::VF32_WIDTH) < accum_len {
            let accum_chunk = S::loadu_ps(&accum[i]);
            let src_chunk = S::loadu_ps(&src[i]);
            let sum_chunk = accum_chunk * src_chunk;
            S::storeu_ps(&mut accum[i], sum_chunk);
            i += S::VF32_WIDTH;
        }
        while i < accum_len {
            accum[i] *= src[i];
            i += 1;
        }
    }
);

simd_runtime_generate!(
    pub fn blend_vectors(accum: &mut [f32], src: &[f32], blend_factor: f32) {
        let accum_len = accum.len();
        if accum_len != src.len() {
            panic!("Vector lengths don't match: {} vs. {}", accum_len, src.len());
        }
        let blend_factor = blend_factor.clamp(0.0, 1.0);
        let blend_chunk_accum = S::loadu_ps(&vec![blend_factor; S::VF32_WIDTH][0]);
        let blend_chunk_src = S::loadu_ps(&vec![1.0 - blend_factor; S::VF32_WIDTH][0]);
        let mut i = 0;
        while (i + S::VF32_WIDTH) < accum_len {
            let accum_chunk = S::loadu_ps(&accum[i]);
            let src_chunk = S::loadu_ps(&src[i]);
            let sum_chunk = blend_chunk_accum * accum_chunk + blend_chunk_src * src_chunk;
            S::storeu_ps(&mut accum[i], sum_chunk);
            i += S::VF32_WIDTH;
        }
        while i < accum_len {
            accum[i] = blend_factor * accum[i] + (1.0 - blend_factor) * src[i];
            i += 1;
        }
    }
);

simd_runtime_generate!(
    pub fn blend_vectors_variable(accum: &mut [f32], src: &[f32], factors: &[f32]) {
        let accum_len = accum.len();
        if accum_len != src.len() || accum_len != factors.len() {
            panic!("Vector lengths don't match: {} vs. {} vs. {}", accum_len, src.len(), factors.len());
        }
        let ones = S::loadu_ps(&vec![1.0; S::VF32_WIDTH][0]);
        let mut i = 0;
        while (i + S::VF32_WIDTH) < accum_len {
            let blend_chunk_accum = S::loadu_ps(&factors[i]);
            let blend_chunk_src = ones - blend_chunk_accum;
            let accum_chunk = S::loadu_ps(&accum[i]);
            let src_chunk = S::loadu_ps(&src[i]);
            let sum_chunk = blend_chunk_accum * accum_chunk + blend_chunk_src * src_chunk;
            S::storeu_ps(&mut accum[i], sum_chunk);
            i += S::VF32_WIDTH;
        }
        while i < accum_len {
            let factor = factors[i];
            accum[i] = factor * accum[i] + (1.0 - factor) * src[i];
            i += 1;
        }
    }
);

simd_runtime_generate!(
    pub fn add_vectors_variable(base: &mut [f32], additive: &[f32], factors: &[f32]) {
        let base_len = base.len();
        if base_len != additive.len() || base_len != factors.len() {
            panic!(
                "Vector lengths don't match: {} vs. {} vs. {}",
                base_len, additive.len(), factors.len(),
            );
        }
        let mut i = 0;
        while (i + S::VF32_WIDTH) < base_len {
            let factors_chunk = S::loadu_ps(&factors[i]);
            let base_chunk = S::loadu_ps(&base[i]);
            let additive_chunk = S::loadu_ps(&additive[i]);
            let sum_chunk = base_chunk + (additive_chunk * factors_chunk);
            S::storeu_ps(&mut base[i], sum_chunk);
            i += S::VF32_WIDTH;
        }
        while i < base_len {
            let factor = factors[i];
            base[i] += factor * additive[i];
            i += 1;
        }
    }
);

#[cfg(test)]
mod simd_tests {
    use super::*;

    #[test]
    fn test_add_vectors_short() {
        let mut v1 = vec![0.0, 1.0];
        let v2 = vec![2.0, 3.0];
        add_vectors_runtime_select(&mut v1, &v2);
        assert_eq!(v1[0], 2.0);
        assert_eq!(v1[1], 4.0);
    }

    #[test]
    fn test_add_vectors_long() {
        let v1_immut = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,  8.0,  9.0];
        let v2 =       vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut v1 = v1_immut.clone();
        add_vectors_runtime_select(&mut v1, &v2);
        for i in 0..v1_immut.len() {
            assert_eq!(v1[i], v1_immut[i] + v2[i]);
        }
    }

    #[test]
    fn test_subtract_vectors_short() {
        let mut v1 = vec![0.0, 1.0];
        let v2 = vec![2.0, 3.0];
        subtract_vectors_runtime_select(&mut v1, &v2);
        assert_eq!(v1[0], -2.0);
        assert_eq!(v1[1], -2.0);
    }

    #[test]
    fn test_subtract_vectors_long() {
        let v1_immut = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,  8.0,  9.0];
        let v2 =       vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut v1 = v1_immut.clone();
        subtract_vectors_runtime_select(&mut v1, &v2);
        for i in 0..v1_immut.len() {
            assert_eq!(v1[i], v1_immut[i] - v2[i]);
        }
    }

    #[test]
    fn test_multiply_vectors_short() {
        let mut v1 = vec![0.0, 1.0];
        let v2 = vec![2.0, 3.0];
        multiply_vectors_runtime_select(&mut v1, &v2);
        assert_eq!(v1[0], 0.0);
        assert_eq!(v1[1], 3.0);
    }

    #[test]
    fn test_multiply_vectors_long() {
        let v1_immut = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,  8.0,  9.0];
        let v2 =       vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut v1 = v1_immut.clone();
        multiply_vectors_runtime_select(&mut v1, &v2);
        for i in 0..v1_immut.len() {
            assert_eq!(v1[i], v1_immut[i] * v2[i]);
        }
    }

    #[test]
    fn test_blend_vectors_short() {
        let mut v1 = vec![0.0, 1.0];
        let v2 = vec![2.0, 3.0];
        blend_vectors_runtime_select(&mut v1, &v2, 0.0);
        assert_eq!(v1[0], 2.0);
        assert_eq!(v1[1], 3.0);

        let mut v1 = vec![0.0, 1.0];
        let v2 = vec![2.0, 3.0];
        blend_vectors_runtime_select(&mut v1, &v2, 1.0);
        assert_eq!(v1[0], 0.0);
        assert_eq!(v1[1], 1.0);
    }

    #[test]
    fn test_blend_vectors_long() {
        let v1_immut = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,  8.0,  9.0];
        let v2 =       vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut v1 = v1_immut.clone();
        let factor = 0.0;
        blend_vectors_runtime_select(&mut v1, &v2, factor);
        for i in 0..v1_immut.len() {
            let blended = v1_immut[i] * factor + v2[i] * (1.0 - factor);
            assert!((v1[i] - blended).abs() < f32::EPSILON);
        }

        let v1_immut = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,  8.0,  9.0];
        let v2 =       vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut v1 = v1_immut.clone();
        let factor = 1.0;
        blend_vectors_runtime_select(&mut v1, &v2, factor);
        for i in 0..v1_immut.len() {
            let blended = v1_immut[i] * factor + v2[i] * (1.0 - factor);
            assert!((v1[i] - blended).abs() < f32::EPSILON);
        }

        let v1_immut = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,  8.0,  9.0];
        let v2 =       vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut v1 = v1_immut.clone();
        let factor = 0.5;
        blend_vectors_runtime_select(&mut v1, &v2, factor);
        for i in 0..v1_immut.len() {
            let blended = v1_immut[i] * factor + v2[i] * (1.0 - factor);
            assert!((v1[i] - blended).abs() < f32::EPSILON);
        }

        let v1_immut = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,  8.0,  9.0, 10.0];
        let v2 =       vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut v1 = v1_immut.clone();
        let factor = 0.5;
        blend_vectors_runtime_select(&mut v1, &v2, factor);
        for i in 0..v1_immut.len() {
            let blended = v1_immut[i] * factor + v2[i] * (1.0 - factor);
            assert!((v1[i] - blended).abs() < f32::EPSILON);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OctaveParams { 
    pub frequency: f32,
    pub amplitude: f32,
}

impl OctaveParams {
    pub fn new(frequency: f32, amplitude: f32) -> Self {
        Self{
            frequency,
            amplitude,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NoiseParams {
    pub octaves: Vec<OctaveParams>,
    pub rand_seed: i32,
    pub width: u32,
    pub height: u32,
}

impl NoiseParams {
    pub fn from_pairs(width: u32, height: u32, rand_seed: i32, pairs: &[(f32, f32)]) -> Self {
        let mut octaves = Vec::with_capacity(pairs.len());
        for (freq, amp) in pairs.iter() {
            let frequency = *freq;
            let amplitude = *amp;
            octaves.push(OctaveParams{
                frequency,
                amplitude,
            });
        }
        Self{
            octaves,
            rand_seed,
            width,
            height,
        }
    }

    pub fn new(width: u32, height: u32, rand_seed: i32, params: &[OctaveParams]) -> Self {
        let mut octaves = Vec::with_capacity(params.len());
        for p in params.iter() {
            octaves.push(*p);
        }
        Self{
            octaves,
            rand_seed,
            width,
            height,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HeightMap {
    width: u32,
    height: u32,
    heights: Vec<f32>,
}

impl HeightMap {
    pub fn new(width: u32, height: u32, init_val: f32) -> Self {
        Self{
            width,
            height,
            heights: vec![init_val; width as usize * height as usize],
        }
    }

    pub fn from_flat(width: u32, height: u32, heights: Vec<f32>) -> Self {
        if heights.len() != width as usize * height as usize {
            panic!("Improper length: {} * {} != {}", width, height, heights.len());
        }
        Self{
            width,
            height,
            heights: heights.iter().copied().collect(),
        }
    }

    pub fn from_noise(params: &NoiseParams) -> Self {
        Self::from_flat(params.width, params.height, gen_wrapped_noise(&params))
    }

    pub fn write<W>(&self, mut dst: W) -> std::io::Result<usize>
    where
        W: std::io::Write,
    {
        let mut bytes_written = 0;
        bytes_written += dst.write(&['W' as u8, 'H' as u8, 'M' as u8, '1' as u8])?;
        bytes_written += dst.write(&self.width.to_le_bytes())?;
        bytes_written += dst.write(&self.height.to_le_bytes())?;
        for h in &self.heights {
            let h_bytes = h.to_le_bytes();
            bytes_written += dst.write(&h_bytes)?;
        }
        Ok(bytes_written)
    }

    pub fn load<R>(mut src: R) -> std::io::Result<Self>
    where
        R: std::io::Read,
    {
        use std::io::Error;
        use std::io::ErrorKind;
        let mut magic_bytes = [0x00, 0x00, 0x00, 0x00];
        src.read(&mut magic_bytes)?;
        if magic_bytes[0] != 'W' as u8 ||
            magic_bytes[1] != 'H' as u8 ||
            magic_bytes[2] != 'M' as u8 ||
            magic_bytes[3] != '1' as u8
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "Invalid magic bytes: {:#x} {:#x} {:#x} {:#x}",
                    magic_bytes[0], magic_bytes[1], magic_bytes[2], magic_bytes[3],
                ),
            ));
        }

        let mut width_bytes = [0x00, 0x00, 0x00, 0x00];
        if src.read(&mut width_bytes)? < 4 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "Could not read 4 bytes for width",
            ));
        }
        let width = u32::from_le_bytes(width_bytes);

        let mut height_bytes = [0x00, 0x00, 0x00, 0x00];
        if src.read(&mut height_bytes)? < 4 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "Could not read 4 bytes for height",
            ));
        }
        let height = u32::from_le_bytes(height_bytes);

        let len = (width * height) as usize;
        let mut data = Vec::with_capacity(len);
        for index in 0..len {
            let mut data_bytes = [0x00, 0x00, 0x00, 0x00];
            if src.read(&mut data_bytes)? < 4 {
                return Err(Error::new(
                    ErrorKind::UnexpectedEof,
                    format!("Could not read 4 bytes for a float value at index {}", index),
                ));
            }
            data.push(f32::from_le_bytes(data_bytes));
        }

        Ok(Self{
            width,
            height,
            heights: data,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn save_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        if filename.ends_with(".bin") {
            use std::fs::*;
            use std::io::*;
            let mut f = File::create(filename)?;
            write!(f, "WHM1")?;
            serde_cbor::to_writer(f, &self)?;
        } else {
            use image::{ImageBuffer, Luma};
            let u16_max = u16::MAX as usize;
            
            let mut img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(self.width as u32, self.height as u32);
            for x in 0..self.width {
                for y in 0..self.height {
                    let val = (self.get(x, y) * u16_max as f32) as usize;
                    let val = val.clamp(0, u16_max) as u16;
                    img.put_pixel(x, y, image::Luma([val]));
                }
            }
            img.save(filename)?;
        }
        Ok(())
    }

    pub fn load_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        if filename.ends_with(".bin") {
            use std::fs::*;
            use std::io::*;
            let mut f = File::open(filename)?;
            let mut magic = [0; 4];
            if f.read(&mut magic)? != 4 {
                return Err(
                    Box::new(std::io::Error::new(
                        ErrorKind::UnexpectedEof,
                        "unable to read the expected 4 magic bytes",
                    ))
                );
            }
            let magic_str = std::str::from_utf8(&magic)?;
            if magic_str != "WHM1" {
                return Err(
                    Box::new(std::io::Error::new(
                        ErrorKind::InvalidInput,
                        "magic bytes are incorrect",
                    ))
                );
            }
            Ok(serde_cbor::from_reader(f)?)
        } else {
            use image::DynamicImage::*;
            let img = image::io::Reader::open(filename)?.decode()?;
            match img {
                ImageLuma8(img) => {
                    let (width, height) = img.dimensions();
                    let mut hm = Self::new(width, height, 0.0);
                    for x in 0..width {
                        for y in 0..height {
                            let v = hm.get_mut(x, y);
                            *v = (img.get_pixel(x, y)[0] as f32 / 255.0).clamp(0.0, 1.0);
                        }
                    }
                    Ok(hm)
                },
                ImageLuma16(img) => {
                    let (width, height) = img.dimensions();
                    let u16_max_f32 = u16::MAX as f32;
                    let mut hm = Self::new(width, height, 0.0);
                    for x in 0..width {
                        for y in 0..height {
                            let v = hm.get_mut(x, y);
                            *v = (img.get_pixel(x, y)[0] as f32 / u16_max_f32).clamp(0.0, 1.0);
                        }
                    }
                    Ok(hm)
                },
                _ => {
                    Err(
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Not an 8-bit or 16-bit grayscale image",
                        ))
                    )
                },
            }
        }
    }

    pub fn iter_mut<'a, 'b>(&'a mut self) -> std::slice::IterMut<'b, f32>
    where
        'a: 'b
    {
        self.heights.iter_mut()
    }

    // Returns (min, max, avg)
    pub fn get_stats(&self) -> (f32, f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0;
        for v in self.heights.iter() {
            min = f32::min(min, *v);
            max = f32::max(max, *v);
            sum += *v;
        }

        (min, max, sum / self.heights.len() as f32)
    }

    pub fn get(&self, x: u32, y: u32) -> f32 {
        let x = x as usize;
        let y = y as usize;
        let width = self.width as usize;
        self.heights[x + y * width]
    }

    pub fn get_mut(&mut self, x: u32, y: u32) -> &mut f32 {
        let x = x as usize;
        let y = y as usize;
        let width = self.width as usize;
        if x + y * width >= self.heights.len() {
            dbg!(x);
            dbg!(y);
            dbg!(self.width);
            dbg!(self.height);
            dbg!(self.heights.len());
            dbg!(x + y * width);
        }
        &mut self.heights[x + y * width]
    }

    // This assumes that x >= -width and x < 2 * width
    fn get_wrapped_x(&self, x: i64) -> usize {
        let width = self.width as i64;
        (if x < 0 {
            x + width
        } else if x >= width {
            x - width
        } else {
            x
        }).clamp(0, width - 1) as usize
    }

    // This assumes that y >= -height and y < 2 * height
    fn get_wrapped_y(&self, y: i64) -> usize {
        let height = self.height as i64;
        (if y < 0 {
            y + height
        } else if y >= height {
            y - height
        } else {
            y
        }).clamp(0, height - 1) as usize
    }

    pub fn get_index_wrapped_x(&self, x: i64, y: i64) -> usize {
        let x = self.get_wrapped_x(x);
        let y = y as usize;
        let width = self.width as usize;
        x + y * width
    }

    pub fn get_value_wrapped_x(&self, x: i64, y: i64) -> f32 {
        let x = self.get_wrapped_x(x);
        let y = y as usize;
        let width = self.width as usize;
        self.heights[x + y * width]
    }

    pub fn add(&mut self, other: &Self) {
        add_vectors_runtime_select(&mut self.heights, &other.heights);
    }

    pub fn blend(&mut self, other: &Self, factor: f32) {
        blend_vectors_runtime_select(&mut self.heights, &other.heights, factor);
    }

    pub fn blend_variable(&mut self, other: &Self, factors: &Self) {
        blend_vectors_variable_runtime_select(&mut self.heights, &other.heights, &factors.heights);
    }

    pub fn add_variable(&mut self, other: &Self, factors: &Self) {
        add_vectors_variable_runtime_select(&mut self.heights, &other.heights, &factors.heights);
    }

    #[allow(clippy::collapsible_else_if)]
    pub fn grad_and_height(&self, pos: Vector2<f32>, wrap_x: bool) -> (Vector2<f32>, f32) {
        let (x_min, x_max) = if wrap_x {
            let x_min = pos.x.floor() as i64;
            let x_max = pos.x.floor() as i64 + 1;
            (self.get_wrapped_x(x_min), self.get_wrapped_x(x_max))
        } else {
            let (x_min, x_max) = if pos.x < 0.0 {
                (0, 1)
            } else if pos.x >= (self.width - 2) as f32 {
                ((self.width - 2) as i64, (self.width - 1) as i64)
            } else {
                let x_min = pos.x.floor() as i64;
                (x_min, x_min + 1)
            };
            (x_min as usize, x_max as usize)
        };
        let (y_min, y_max) = {
            let (y_min, y_max) = if pos.y < 0.0 {
                (0, 1)
            } else if pos.y >= (self.height - 2) as f32 {
                ((self.height - 2) as i64, (self.height - 1) as i64)
            } else {
                let y_min = pos.y.floor() as i64;
                (y_min, y_min + 1)
            };
            (y_min as usize, y_max as usize)
        };

        let width = self.width as usize;

        let u = pos.x - pos.x.floor();
        let v = pos.y - pos.y.floor();

        let i_min_min = x_min + y_min * width;
        let i_min_max = x_min + y_max * width;
        let i_max_min = x_max + y_min * width;
        let i_max_max = x_max + y_max * width;

        let p_min_min = self.heights[i_min_min];
        let p_min_max = self.heights[i_min_max];
        let p_max_min = self.heights[i_max_min];
        let p_max_max = self.heights[i_max_max];

        (
            Vector2::new(
                (p_max_min - p_min_min) * (1.0 - v) + (p_max_max - p_min_max) * v,
                (p_min_max - p_min_min) * (1.0 - u) + (p_max_max - p_max_min) * u,
            ),
            p_min_min * (1.0 - u) * (1.0 - v) +
                p_max_min * u * (1.0 - v) +
                p_min_max * (1.0 - u) * v +
                p_max_max * u * v,
        )
    }

    pub fn deposit(&mut self, pos: Vector2<f32>, amount: f32, wrap_x: bool) {
        let x_min = if wrap_x {
            self.get_wrapped_x(pos.x.floor() as i64) as u32
        } else {
            pos.x.floor() as u32
        };
        let x_max = if wrap_x {
            self.get_wrapped_x(pos.x.floor() as i64 + 1) as u32
        } else {
            x_min + 1
        };
        let y_min = pos.y.floor() as u32;
        let y_max = y_min + 1;

        let u = pos.x - pos.x.floor();
        let v = pos.y - pos.y.floor();

        let x_min = x_min as usize;
        let x_max = x_max as usize;
        let y_min = y_min as usize;
        let y_max = y_max as usize;
        let width = self.width as usize;

        let i_min_min = x_min + y_min * width;
        let i_min_max = x_min + y_max * width;
        let i_max_min = x_max + y_min * width;
        let i_max_max = x_max + y_max * width;

        if i_min_min < self.heights.len() {
            self.heights[i_min_min] += amount * (1.0 - u) * (1.0 - v);
        }
        if i_max_min < self.heights.len() {
            self.heights[i_max_min] += amount * u * (1.0 - v);
        }
        if i_min_max < self.heights.len() {
            self.heights[i_min_max] += amount * (1.0 - u) * v;
        }
        if i_max_max < self.heights.len() {
            self.heights[i_max_max] += amount * u * v;
        }
    }

    // This function assumes that the range is not wider than the map.
    // It returns indices and distances.
    pub fn measure_wrapped_x(
        &self,
        pos: Vector2<f32>,
        radius: f32,
    ) -> Vec<(usize, f32)> {
        let (min_x, max_x, min_y, max_y) = {
            let radius_int = radius as i64;
            let pos_x = pos.x as i64;
            let pos_y = pos.y as i64;
            let min_x = i64::max(pos_x - radius_int, 0);
            let max_x = i64::min(pos_x + radius_int, (self.width - 1) as i64);
            let min_y = i64::max(pos_y - radius_int, 0);
            let max_y = i64::min(pos_y + radius_int, (self.height - 1) as i64);

            // Account for wrapping of the x coordinate
            let min_x = self.get_wrapped_x(min_x);
            let max_x = self.get_wrapped_x(max_x);
            // Bail out early if we have a degenerate y range
            if max_y < min_y {
                return Vec::new();
            }
            (min_x, max_x, min_y, max_y)
        };
        let radius2 = radius * radius;

        let mut indices = Vec::new();
        if min_x < max_x {
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let dx = x as f32 - pos.x;
                    let dy = y as f32 - pos.y;
                    let dist2 = dx * dx + dy * dy;
                    if dist2 <= radius2 {
                        let dist = f32::sqrt(dist2);
                        let i = x as usize + y as usize * self.width as usize;
                        if i >= self.heights.len() {
                            dbg!(min_y);
                            dbg!(max_y);
                            dbg!(i);
                            dbg!(x);
                            dbg!(y);
                            dbg!(self.width);
                            dbg!(self.height);
                            panic!("BAD!");
                        }
                        indices.push((i, dist));
                    }
                }
            }
        } else {
            let width_f32 = self.width as f32;
            let pos1 = pos;
            // In both of these cases, if the point is in bounds, the new point will be out of
            // bounds.  If it's out of bounds, the new point will be in bounds.  In either case,
            // it should be in the correct position with respect to the wrapped points.
            let pos2 = if pos.x < width_f32 / 2.0 {
                // We're on the left side of the map, so put the other point off the right side
                Vector2::new(pos.x + width_f32, pos.y)
            } else {
                // We're on the right side of the map, so put the other point off the left side
                Vector2::new(pos.x - width_f32, pos.y)
            };
            for y in min_y..=max_y {
                for x in (0..=max_x).chain(min_x..(self.width as usize)) {
                    let ppos = Vector2::new(x as f32, y as f32);
                    let d1 = ppos - pos1;
                    let d2 = ppos - pos2;
                    // We can avoid one square root calculation by comparing the squared
                    // distances (a^2 < b^2 ==> a < b) and then computing the square root
                    // of the resulting value.
                    let dist2 = f32::min(d1.magnitude_squared(), d2.magnitude_squared());
                    if dist2 <= radius2 {
                        let dist = f32::sqrt(dist2);
                        let i = x as usize + y as usize * self.width as usize;
                        if i >= self.heights.len() {
                            dbg!(i);
                            dbg!(x);
                            dbg!(y);
                            dbg!(self.width);
                            dbg!(self.height);
                            panic!("BAD!");
                        }
                        indices.push((i, dist));
                    }
                }
            }
        }
        indices
    }

    pub fn apply_erosion(&mut self, pos: Vector2<f32>, radius: f32, amount: f32, wrap_x: bool) -> f32 {
        // i64 is used here so that we can address an entire u32::MAX by u32::MAX space.
        let radius_int = radius.ceil() as i64;
        let radius2 = radius * radius;
        let pos_x = pos.x.floor() as i64;
        let pos_y = pos.y.floor() as i64;
        let points = if wrap_x {
            self.measure_wrapped_x(pos, radius)
        } else {
            let min_x = i64::max(pos_x - radius_int, 0);
            let max_x = i64::min(pos_x + radius_int, self.width as i64);
            let min_y = i64::max(pos_y - radius_int, 0);
            let max_y = i64::min(pos_y + radius_int, self.height as i64);
            let mut points = Vec::with_capacity(((max_x - min_x) * (max_y - min_y)) as usize);
            for x in min_x..max_x {
                for y in min_y..max_y {
                    let dx = pos.x - x as f32;
                    let dy = pos.y - y as f32;
                    let dist2 = dx * dx + dy * dy;
                    if dist2 <= radius2 {
                        let dist = f32::sqrt(dist2);
                        let i = x + y * self.width as i64;
                        let w = f32::max(0.0, radius - dist);
                        if i > 0 {
                            points.push((i as usize, w));
                        }
                    }
                }
            }
            points
        };
        let mut w_sum = 0.0;
        for (_, w) in points.iter() {
            w_sum += *w;
        }
        let mut total_eroded = 0.0;
        for (i, w) in points.iter() {
            let i = *i as usize;
            let amount_eroded = amount * (*w / w_sum);
            total_eroded += amount_eroded;
            self.heights[i] = f32::max(
                0.0,
                self.heights[i] - amount_eroded,
            );
        }
        total_eroded
    }

    pub fn normalize(&mut self) {
        // TODO: change this to a call to constrain() instead.
        let mut current_min = f32::MAX;
        for v in self.heights.iter() {
            current_min = f32::min(current_min, *v);
        }
        let mut current_max = f32::MIN;
        for v in self.heights.iter_mut() {
            *v -= current_min;
            if *v < 0.0 {
                *v = 0.0;
            }
            current_max = f32::max(current_max, *v);
        }
        let scale = 1.0 / current_max;
        for v in self.heights.iter_mut() {
            *v *= scale;
            if *v > 1.0 {
                *v = 1.0;
            }
        }
    }

    pub fn erode(&mut self, config: &ErosionConfig, rand_seed: u64, wrap_x: bool, mask: &Self) {
        if mask.heights.len() != self.heights.len() {
            panic!("Improper mask size: {} vs. {}", mask.heights.len(), self.heights.len());
        }
        /*let mut avg_lifetime = 0;
        let mut num_in_oceans = 0;
        let mut num_out_of_bounds = 0;
        let mut num_stopped = 0;*/
        let mut rng = SmallRng::seed_from_u64(rand_seed);
        let width_f32 = self.width as f32;
        let height_f32 = self.height as f32;
        let backup = Self{
            heights: self.heights.clone(),
            width: self.width,
            height: self.height,
        };
        for _ in 0..config.num_droplets {
            let mut droplet = ErosionDroplet{
                pos: Vector2::new(
                    rng.gen_range(0.0..width_f32),
                    rng.gen_range(0.0..height_f32),
                ),
                dir: Vector2::new(
                    rng.gen_range(-1.0..=1.0),
                    rng.gen_range(-1.0..=1.0),
                ).normalize(),
                vel: config.initial_speed,
                water: config.initial_water,
                sediment: 0.0,
            };
            let mut max_tick = 0;
            for tick in 0..config.max_droplet_lifetime {
                max_tick = tick;
                let (grad, old_height) = self.grad_and_height(droplet.pos, wrap_x);
                //dbg!((grad.x, grad.y));
                //dbg!((droplet.dir.x, droplet.dir.y));

                droplet.dir = droplet.dir * config.inertia + grad * (1.0 - config.inertia);
                if droplet.dir.x.abs() < f32::EPSILON && droplet.dir.y.abs() < f32::EPSILON {
                    droplet.dir = Vector2::new(
                        rng.gen_range(-1.0..=1.0),
                        rng.gen_range(-1.0..=1.0),
                    );
                }
                droplet.dir.normalize_mut();
                //dbg!((droplet.dir.x, droplet.dir.y));

                //dbg!((droplet.pos.x, droplet.pos.y));
                droplet.pos += droplet.dir;
                //dbg!((droplet.pos.x, droplet.pos.y));

                let out_of_bounds = if wrap_x {
                    if droplet.pos.x < -config.erosion_radius {
                        droplet.pos.x += width_f32;
                    } else if droplet.pos.x >= width_f32 + config.erosion_radius {
                        droplet.pos.x -= width_f32;
                    }
                    droplet.pos.y < 0.0 || droplet.pos.y >= height_f32
                } else {
                    droplet.pos.x < 0.0 || droplet.pos.x >= width_f32 ||
                        droplet.pos.y < 0.0 || droplet.pos.y >= height_f32
                };
                let out_of_bounds = out_of_bounds || {
                    let mask_sample = mask.sample_bilinear(droplet.pos);
                    /*if mask_sample < 0.5 {
                        num_in_oceans += 1;
                    }*/
                    // If the mask value is less than 0.5 (half the range),
                    // consider the droplet to be gone.
                    mask_sample < 0.5
                };

                /*if out_of_bounds {
                    num_out_of_bounds += 1;
                }

                if droplet.dir.x.abs() < f32::EPSILON && droplet.dir.y.abs() < f32::EPSILON {
                    num_stopped += 1;
                }*/

                if (droplet.dir.x.abs() < f32::EPSILON && droplet.dir.y.abs() < f32::EPSILON) ||
                    out_of_bounds {
                        // Droplet has stopped or is out of bounds
                        break;
                    }
                let (_, new_height) = self.grad_and_height(droplet.pos, wrap_x);
                let delta_height = new_height - old_height + 0.001;
                let sediment_capacity = f32::max(
                    -delta_height * droplet.vel * droplet.water * config.sediment_capacity_factor,
                    config.min_sediment_capacity,
                );
                if droplet.sediment > sediment_capacity || delta_height > 0.0 {
                    // Droplet is moving uphill or has exceeded its sediment capacity
                    let amount_to_deposit = if delta_height > 0.0 {
                        f32::min(delta_height, droplet.sediment)
                    } else {
                        (droplet.sediment - sediment_capacity) * config.deposit_speed
                    };
                    droplet.sediment -= amount_to_deposit;
                    self.deposit(droplet.pos, amount_to_deposit, wrap_x);
                } else {
                    // Droplet is moving downhill or can carry more sediment
                    let amount_to_erode = f32::min(
                        (sediment_capacity - droplet.sediment) * config.erode_speed,
                        -delta_height,
                    );
                    let amount_eroded = self.apply_erosion(
                        droplet.pos,
                        config.erosion_radius,
                        amount_to_erode,
                        wrap_x,
                    );
                    droplet.sediment -= amount_eroded;
                }

                droplet.vel = f32::sqrt(droplet.vel * droplet.vel + delta_height * config.gravity);
                droplet.water *= 1.0 - config.evaporate_speed;
            }
            //avg_lifetime += max_tick;
        }
        /*let avg_lifetime = avg_lifetime as f32 / config.num_droplets as f32;
        dbg!(avg_lifetime);
        let frac_in_oceans = num_in_oceans as f32 / config.num_droplets as f32;
        dbg!(frac_in_oceans);
        let frac_out_of_bounds = num_out_of_bounds as f32 / config.num_droplets as f32;
        dbg!(frac_out_of_bounds);
        let frac_stopped = num_stopped as f32 / config.num_droplets as f32;
        dbg!(frac_stopped);*/
        let mut changes = Self{
            heights: self.heights.clone(),
            width: self.width,
            height: self.height,
        };
        self.heights = backup.heights;
        subtract_vectors_runtime_select(&mut changes.heights, &self.heights);
        changes.gaussian_blur(2.0);
        add_vectors_runtime_select(&mut self.heights, &changes.heights);
    }

    pub fn gaussian_blur(&mut self, radius: f32) {
        let (kernel, offsets) = {
            let effect_threshold = 0.005;
            let kernel_size = (
                1.0 + 2.0 *
                    f32::sqrt(
                        -2.0 * radius * radius * f32::ln(effect_threshold)
                    ))
                .floor() + 1.0;
            let kernel_size = kernel_size as usize;
            dbg!(kernel_size);
            let mut kernel = vec![0.0; kernel_size];
            let mut offsets = vec![0; kernel_size];
            let twice_radius_squared_recip = 1.0 / (2.0 * radius * radius);
            let sqrt_two_pi_times_radius_recip = 1.0 / (f32::sqrt(2.0 * PI) * radius);
            let radius_modifier = 1.0;

            let center = kernel_size / 2;
            let mut sum = 0.0;
            for i in 0..kernel_size {
                let x = (i as i32 - center as i32) as f32;
                kernel[i] = gaussian_simpson_integration(radius, x - 0.5, x + 0.5);
                offsets[i] = i as i64 - center as i64;
                sum += kernel[i];
            }

            for v in kernel.iter_mut() {
                *v /= sum;
            }

            (kernel, offsets)
        };

        // Blur horizontally from self into tmp
        let mut tmp = Self{
            width: self.width,
            height: self.height,
            heights: vec![0.0; self.width as usize * self.height as usize],
        };
        for y in 0..(self.height) {
            for x in 0..(self.width as i64) {
                let mut sum = 0.0;
                for (i, v) in offsets.iter().enumerate() {
                    sum += self.get(self.get_wrapped_x(x + *v) as u32, y) * kernel[i];
                }
                *tmp.get_mut(x as u32, y) = sum;
            }
        }

        // Blur vertically from tmp into self
        self.heights = vec![0.0; self.width as usize * self.height as usize];
        for y in 0..(tmp.height as i64) {
            for x in 0..(tmp.width as u32) {
                let mut sum = 0.0;
                for (i, v) in offsets.iter().enumerate() {
                    sum += tmp.get(x, self.get_wrapped_y(y + *v) as u32) * kernel[i];
                }
                *self.get_mut(x, y as u32) = sum;
            }
        }
    }

    // warp_x and warp_y are values to warp the sampling of self by
    // So, if you are at (x, y) in the new version, you will see the
    // value from (x + warp_x(x, y), y + warp_y(x, y)) in the old version.
    // warp_x and warp_y should not have values outside the range -1.0..=1.0.
    pub fn warp(&mut self, warp_x: &Self, warp_y: &Self) {
        if warp_x.heights.len() != self.heights.len() {
            panic!("Size mismatch for min(): {} vs. {}", warp_x.heights.len(), self.heights.len());
        }
        if warp_y.heights.len() != self.heights.len() {
            panic!("Size mismatch for min(): {} vs. {}", warp_y.heights.len(), self.heights.len());
        }
        let mut tmp = self.clone();
        for y in 0..(self.height as u32) {
            for x in 0..(self.width as u32) {
                let x_ratio = x as f32 / (self.width - 1) as f32;
                let y_ratio = y as f32 / (self.height - 1) as f32;
                let v = tmp.get_mut(x, y);
                let x_warp_value = warp_x.get(x, y).clamp(-1.0 + f32::EPSILON, 1.0 - f32::EPSILON);
                let y_warp_value = warp_y.get(x, y).clamp(-1.0 + f32::EPSILON, 1.0 - f32::EPSILON);
                *v = self.sample(Vector2::new(
                    x_ratio as f32 + x_warp_value,
                    y_ratio as f32 + y_warp_value,
                ));
            }
        }
        self.heights = tmp.heights;
    }

    // Normalizes, then subtracts from 1.0
    pub fn invert(&mut self) {
        self.normalize();
        for v in self.heights.iter_mut() {
            *v = 1.0 - *v;
        }
    }

    pub fn clamp(&mut self, min: f32, max: f32) {
        for v in self.heights.iter_mut() {
            *v = v.clamp(min, max);
        }
    }

    pub fn cropped(&mut self, min_x: u32, min_y: u32, width: u32, height: u32) -> Self {
        if min_x >= self.width || min_y >= self.height {
            return Self::new(1, 1, f32::NAN);
        }

        let max_x = u32::min(min_x + width, self.width);
        let max_y = u32::min(min_y + height, self.height);
        let mut result = Self::new(width, height, 0.0);
        for y in min_y..max_y {
            let col = y - min_y;
            for x in min_x..max_x {
                let row = x - min_x;
                *result.get_mut(row, col) = self.get(x, y);
            }
        }
        result
    }

    pub fn scale(&mut self, max: f32) {
        let mut current_max = 0.0;
        for v in self.heights.iter() {
            current_max = f32::max(current_max, *v);
        }
        let scale = max / current_max;
        for v in self.heights.iter_mut() {
            *v *= scale;
        }
        // DEBUG
        for v in self.heights.iter() {
            if *v > max {
                panic!("{} > {}!", *v, max);
            }
        }
    }

    pub fn scaled_dimensions(&mut self, new_width: u32, new_height: u32) -> Self {
        let mut new_hm = Self::new(new_width, new_height, 0.0);
        for y in 0..new_height {
            let y_ratio = y as f32 / (new_height - 1) as f32;
            let y_ratio = y_ratio.clamp(0.0, 1.0);
            for x in 0..new_width {
                let x_ratio = x as f32 / (new_width - 1) as f32;
                let x_ratio = x_ratio.clamp(0.0, 1.0);
                *new_hm.get_mut(x, y) = self.sample(Vector2::new(x_ratio, y_ratio));
            }
        }

        new_hm
    }

    pub fn constrain(&mut self, min: f32, max: f32) {
        let mut current_min = f32::MAX;
        for v in self.heights.iter() {
            current_min = f32::min(current_min, *v);
        }
        let adj = min - current_min;
        let mut current_max = f32::MIN;
        for v in self.heights.iter_mut() {
            *v += adj;
            if *v < min {
                *v = min;
            }
            current_max = f32::max(current_max, *v);
        }
        if current_max - min != 0.0 {
            let scale = (max - min) / (current_max - min);
            for v in self.heights.iter_mut() {
                let dist = *v - min;
                let dist = dist * scale;
                *v = min + dist;
                if *v > max {
                    *v = max;
                }
            }
        }
    }

    pub fn min(&mut self, other: &Self) {
        if other.heights.len() != self.heights.len() {
            panic!("Size mismatch for min(): {} vs. {}", other.heights.len(), self.heights.len());
        }

        for (i, s) in self.heights.iter_mut().enumerate() {
            let o = other.heights[i];
            *s = f32::min(*s, o);
        }
    }

    // Adds the specified amount to every heightmap value
    pub fn elevate(&mut self, amount: f32) {
        for v in self.heights.iter_mut() {
            *v += amount;
        }
    }

    pub fn noisify(&mut self, noise_scale: f32, rand_seed: i32) {
        let mut noise = Self::new(self.width, self.height, 0.0);
        for (freq, amp) in &[(1.0, 0.1), (2.0, 0.05), (4.0, 0.025), (8.0, 0.0125)] {
            let octave = NoiseBuilder::fbm_2d(self.width as usize, self.height as usize)
                .with_freq(*freq / 40.0)
                .with_seed(rand_seed)
                .generate_scaled(0.0, *amp * noise_scale);
            let octave = Self::from_flat(self.width, self.height, octave);
            noise.add(&octave);
        }
        let mut one = vec![1.0; self.width as usize * self.height as usize];
        subtract_vectors_runtime_select(&mut one, &noise.heights);
        multiply_vectors_runtime_select(&mut self.heights, &one);
    }

    // Samples like OpenGL would, using 0.0..=1.0 as the range for x and y.
    pub fn sample(&self, pos: Vector2<f32>) -> f32 {
        self.sample_bilinear(
            Vector2::new(
                pos.x * self.width as f32,
                pos.y * self.height as f32,
            )
        )
    }

    // Wraps x and y.
    fn sample_bilinear(&self, pos: Vector2<f32>) -> f32 {
        let u = pos.x - pos.x.floor();
        let v = pos.y - pos.y.floor();
        let x_min = pos.x.floor();
        let y_min = pos.y.floor();

        let v_min_min = {
            let x_min = self.get_wrapped_x(x_min as i64);
            let y_min = self.get_wrapped_y(y_min as i64);
            let width = self.width as usize;
            self.heights[x_min + width * y_min]
        };
        let v_min_max = {
            let x_min = self.get_wrapped_x(x_min as i64);
            let y_max = self.get_wrapped_y(y_min as i64 + 1);
            let width = self.width as usize;
            self.heights[x_min + width * y_max]
        };
        let v_max_min = {
            let x_max = self.get_wrapped_x(x_min as i64 + 1);
            let y_min = self.get_wrapped_y(y_min as i64);
            let width = self.width as usize;
            self.heights[x_max + width * y_min]
        };
        let v_max_max = {
            let x_max = self.get_wrapped_x(x_min as i64 + 1);
            let y_max = self.get_wrapped_y(y_min as i64 + 1);
            let width = self.width as usize;
            self.heights[x_max + width * y_max]
        };

        v_min_min * (1.0 - u) * (1.0 - v) +
            v_max_min * u * (1.0 - v) +
            v_min_max * (1.0 - u) * v +
            v_max_max * u * v
    }

    pub fn add_item(&mut self, other: &Self, pos: Vector2<f32>, h_size: f32, v_scale: f32) {
        let scaled_width = h_size.ceil() as usize;
        let scaled_height = (h_size * (other.height as f32) / (other.width as f32)).ceil() as usize;
        let min_x = pos.x.floor() as usize;
        let min_y = pos.y.floor() as usize;
        let max_x = min_x + scaled_width;
        let max_y = min_y + scaled_height;
        for x_dst in min_x..max_x {
            for y_dst in min_y..max_y {
                if y_dst < self.height as usize {
                    let x_src = other.width as f32 * ((x_dst - min_x) as f32) / (scaled_width as f32);
                    let y_src = other.height as f32 * ((y_dst - min_y) as f32) / (scaled_height as f32);
                    let pos_src = Vector2::new(x_src, y_src);
                    let src_val = other.sample_bilinear(pos_src);
                    let x_dst = x_dst % self.width as usize;
                    self.heights[x_dst + y_dst * self.width as usize] += src_val * v_scale;
                }
            }
        }
    }

    pub fn blend_item(
        &mut self, other: &Self,
        pos: Vector2<f32>,
        h_size: f32,
        height: f32,
    ) {
        let scaled_width = h_size.ceil() as usize;
        let scaled_height = (h_size * (other.height as f32) / (other.width as f32)).ceil() as usize;
        let min_x = pos.x.floor() as usize;
        let min_y = pos.y.floor() as usize;
        let max_x = min_x + scaled_width;
        let max_y = min_y + scaled_height;
        for x_dst in min_x..max_x {
            for y_dst in min_y..max_y {
                if y_dst < self.height as usize {
                    let x_src = other.width as f32 * ((x_dst - min_x) as f32) / (scaled_width as f32);
                    let y_src = other.height as f32 * ((y_dst - min_y) as f32) / (scaled_height as f32);
                    let pos_src = Vector2::new(x_src, y_src);
                    //let src_val = other.sample_bilinear(pos_src);
                    let factor = other.sample_bilinear(pos_src);
                    let x_dst = x_dst % self.width as usize;
                    let v = &mut self.heights[x_dst + y_dst * self.width as usize];
                    //*v = (src_val * v_scale) * factor + *v * (1.0 - factor);
                    *v = height * factor + *v * (1.0 - factor);
                }
            }
        }
    }

    pub fn apply_bands(&mut self, bands: &Bands) {
        for y in 0..(self.height as usize) {
            for x in 0..(self.width as usize) {
                let i = x + y * self.width as usize;
                let v = &mut self.heights[i];
                *v = bands.get_value(*v);
            }
        }
    }

    pub fn blend_layer(
        &mut self,
        regions_noise: &Self,
        layer_noise: &Self,
    ) {
        self.blend_variable(layer_noise, regions_noise);
    }

    pub fn add_layer(
        &mut self,
        regions_noise: &Self,
        layer_noise: &Self,
    ) {
        self.add_variable(layer_noise, regions_noise);
    }

    // Converts the heightmap (assuming Gall stereographic projection ranging
    // from 85 degrees north to 85 degrees south) to a lookup table for spherical
    // coordinates (x = theta, y = phi, value = rho).
    // pixels_per_degree: how many pixels does the lookup table have per degree?
    // polar_exclusion_zone: size of the flat areas at the poles in degrees
    // polar_transition_zone: width of the transition band between normal terrain and pole in degrees
    pub fn to_spherical(
        &self,
        pixels_per_degree: u32,
        polar_exclusion_zone: f32,
        polar_transition_zone: f32,
        rand_seed: i32,
    ) -> Self {
        let mut polar_noise = gen_wrapped_noise(&NoiseParams::from_pairs(
            360 * pixels_per_degree, 1,
            rand_seed,
            &[
                (1.0 * 80.0, 0.2),
                (2.0 * 80.0, 0.2 / 2.0),
                (4.0 * 80.0, 0.2 / 4.0),
                (8.0 * 80.0, 0.2 / 8.0),
            ],
        ));
        let polar_noise_resolution = polar_noise.len() as f32;
        let mut polar_noise_avg = 0.0;
        for v in polar_noise.iter() {
            polar_noise_avg += *v;
        }
        polar_noise_avg /= polar_noise_resolution;
        dbg!(polar_noise_avg);
        for v in polar_noise.iter_mut() {
            *v = 1.0 + (*v - polar_noise_avg);
        }

        let mut north_pole_avg = 0.0;
        for x in 0..self.width {
            north_pole_avg += self.get(x, 0);
        }
        north_pole_avg /= self.width as f32;

        let mut south_pole_avg = 0.0;
        for x in 0..self.width {
            south_pole_avg += self.get(x, self.height - 1);
        }
        south_pole_avg /= self.width as f32;

        let spherical_width = 360 * pixels_per_degree;
        let spherical_height = 180 * pixels_per_degree;
        let mut spherical = HeightMap::new(
            spherical_width,
            spherical_height,
            0.0,
        );
        let sqrt2 = f32::sqrt(2.0);
        let y_min_limit = (1.0 + sqrt2 / 2.0) * f32::tan(f32::to_radians(-85.0 / 2.0));
        let y_max_limit = (1.0 + sqrt2 / 2.0) * f32::tan(f32::to_radians(85.0 / 2.0));
        let y_range = y_max_limit - y_min_limit;
        let x_min_limit = -180.0 / sqrt2;
        let x_max_limit = 180.0 / sqrt2;
        let x_range = x_max_limit - x_min_limit;
        for theta in 0..spherical_width {
            for phi in 0..spherical_height {
                let long = theta as f32 / 20.0 - 180.0;
                let lat = phi as f32 / 20.0 - 90.0;
                // Hm...my latitudes seem to be backwards.
                // I guess that's just the inevitable result of the coordinate
                // system I'm using.  I'll just ignore that for now...
                let polar_height = if lat < 0.0 {
                    north_pole_avg
                } else {
                    south_pole_avg
                };
                let polar_noise_idx = (theta as f32 / spherical_width as f32) * polar_noise_resolution;
                let polar_noise_sample = polar_noise[polar_noise_idx as usize];
                let polar_exclusion_zone = polar_exclusion_zone * polar_noise_sample;
                let polar_effect_zone = polar_exclusion_zone + polar_transition_zone;
                let v = spherical.get_mut(theta, phi);
                *v = if lat.abs() >= (90.0 - polar_exclusion_zone) {
                    polar_height
                } else if lat.abs() >= (90.0 - polar_effect_zone) {
                    // We want the factor to be:
                    //   0.0 at 90.0 - polar_effect_zone
                    //   1.0 at 90.0 - polar_exclusion_zone
                    let offset = lat.abs() as f32 - (90.0 - polar_effect_zone);
                    // factor should be the fraction of the distance from 90.0 - polar_effect_zone
                    // to 90.0 - polar_exclusion_zone that the current lattitude is.
                    let factor = offset / polar_transition_zone;

                    let x = long / sqrt2;
                    let x = x - x_min_limit;
                    let x_frac = x / x_range;
                    let y = (1.0 + sqrt2 / 2.0) * f32::tan(f32::to_radians(lat / 2.0));
                    let y = y - y_min_limit;
                    let y_frac = y / y_range;
                    let terrain_height = self.sample(Vector2::new(x_frac, y_frac));
                    polar_height * factor + terrain_height * (1.0 - factor)
                } else {
                    let x = long / sqrt2;
                    let x = x - x_min_limit;
                    let x_frac = x / x_range;
                    let y = (1.0 + sqrt2 / 2.0) * f32::tan(f32::to_radians(lat / 2.0));
                    let y = y - y_min_limit;
                    let y_frac = y / y_range;
                    self.sample(Vector2::new(x_frac, y_frac))
                };
            }
        }

        spherical
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Band {
    pos: f32,
    radius: f32,
}

#[derive(Debug, Clone)]
pub struct Bands {
    bands: Vec<Band>,
    num_bands: usize,
    range_min: f32,
    range_max: f32,
    band_width: f32,
}

impl Bands {
    pub fn new(num_bands: usize, range_min: f32, range_max: f32, falloff: f32) -> Self {
        let range = range_max - range_min;
        let band_width = range / (num_bands / 2) as f32;
        let mut bands = Vec::new();
        for i in 0..num_bands {
            let pos = range_min + i as f32 * band_width;
            let radius = band_width * 0.5 * falloff;
            bands.push(Band{
                pos,
                radius,
            });
        }
        for band in bands.iter() {
            dbg!(band.pos);
            dbg!(band.radius);
        }
        dbg!(band_width);
        Self{
            bands,
            num_bands,
            range_min,
            range_max,
            band_width,
        }
    }

    fn get_value(&self, input: f32) -> f32 {
        if input < self.bands[0].pos {
            return self.bands[0].pos;
        } else if input > self.bands[self.num_bands - 1].pos {
            return self.bands[self.num_bands - 1].pos;
        }
        let offset = input - self.range_min;
        let offset = f32::min(f32::max(offset, 0.0), self.range_max);
        let band_idx = (offset / self.band_width).round() as usize;
        let band = &self.bands[band_idx];
        let dist = (input - band.pos).abs();
        if dist < band.radius {
            band.pos
        } else {
            let other_band = if input > band.pos {
                &self.bands[band_idx + 1]
            } else {
                if band_idx == 0 {
                    dbg!(input);
                    dbg!(offset);
                    dbg!(band_idx);
                    dbg!(dist);
                    panic!("SHIT!");
                }
                &self.bands[band_idx - 1]
            };
            let factor = dist / (other_band.pos - band.pos).abs();
            band.pos * (1.0 - factor) + other_band.pos * factor
        }
    }
}

#[cfg(test)]
mod heightmap_tests {
    use super::*;

    #[test]
    fn test_sample_bilinear() {
        let hm = HeightMap{
            width: 2,
            height: 2,
            heights: vec![0.0, 1.0, 1.0, 0.0],
        };

        let pos = Vector2::new(0.5, 0.5);
        let v = hm.sample_bilinear(pos);
        dbg!(pos);
        dbg!(v);
        assert!((v - 0.5).abs() < f32::EPSILON);

        println!("================================================================================");

        let hm = HeightMap{
            width: 2,
            height: 2,
            heights: vec![0.0, 1.0, 1.0, 0.0],
        };

        let pos = Vector2::new(0.0, 0.0);
        let v = hm.sample_bilinear(pos);
        dbg!(pos);
        dbg!(v);
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CraterConfig {
    pub noise_scale: f32,
    pub base_radius: f32,
    pub base_variability: f32,
    pub rim_radius: f32,
    pub rim_variability: f32,
    pub excavation_depth: f32,
    pub fill_height: f32,
    pub erosion: Option<ErosionConfig>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum MountainType {
    Cone(ConeConfig),
    Volcano(VolcanoConfig),
    Chunk(ChunkConfig),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConeConfig {
    pub base_radius: f32,
    pub base_variability: f32,
    pub peak_radius: f32,
    pub peak_variability: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VolcanoConfig {
    pub base_radius: f32,
    pub base_variability: f32,
    pub peak_radius: f32,
    pub peak_variability: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChunkConfig {
    pub base_radius: f32,
    pub base_variability: f32,
    pub peak_radius: f32,
    pub peak_variability: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MountainConfig {
    pub noise_scale: f32,
    pub erosion: Option<ErosionConfig>,
    pub mountain: MountainType,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ErosionConfig {
    pub inertia: f32,
    pub sediment_capacity_factor: f32,
    pub min_sediment_capacity: f32,
    pub erode_speed: f32,
    pub deposit_speed: f32,
    pub evaporate_speed: f32,
    pub gravity: f32,
    pub max_droplet_lifetime: usize,
    pub num_droplets: usize,
    pub erosion_radius: f32,
    pub initial_water: f32,
    pub initial_speed: f32,
}

struct ErosionDroplet {
    pos: Vector2<f32>,
    dir: Vector2<f32>,
    vel: f32,
    water: f32,
    sediment: f32,
}

pub fn gen_wrapped_noise(params: &NoiseParams) -> Vec<f32> {
    let width = params.width;
    let height = params.height;
    let width_bigger = width * 3 / 2;
    let width_half = width / 2;
    let vector_size_bigger = width_bigger as usize * height as usize;
    let mut noise2d: Vec<f32> = vec![0.0; vector_size_bigger];
    for OctaveParams{frequency, amplitude} in &params.octaves {
        let freq = *frequency;
        let amp = *amplitude;
        let more_noise = NoiseBuilder::fbm_2d(width_bigger as usize, height as usize)
            .with_freq(freq / (width as f32 * 2.0 * PI))
            .with_seed(params.rand_seed)
            .generate_scaled(0.0, amp);
        add_vectors_runtime_select(&mut noise2d, &more_noise);
    }

    let vector_size = width as usize * height as usize;
    let mut result = vec![0.0; vector_size];
    for y in 0..(height as usize) {
        for x in 0..(width as usize) {
            let i = x + y * width as usize;
            result[i] = if x < width_half as usize {
                let i_bigger_left = x + y * width_bigger as usize;
                let i_bigger_right = (x + width as usize) + y * width_bigger as usize;
                let factor = x as f32 / width_half as f32;
                let left = noise2d[i_bigger_left];
                let right = noise2d[i_bigger_right];
                left * factor + right * (1.0 - factor)
            } else {
                let i_bigger = x + y * width_bigger as usize;
                noise2d[i_bigger]
            };
        }
    }
    result
}

fn gaussian(sigma: f32, x: f32) -> f32 {
    f32::exp(-(x * x) / (2.0 * sigma * sigma))
}

fn gaussian_simpson_integration(sigma: f32, a: f32, b: f32) -> f32 {
    ((b - a) / 6.0) *
        (gaussian(sigma, a) + 4.0 * gaussian(sigma, (a + b) / 2.0) + gaussian(sigma, b))
}
