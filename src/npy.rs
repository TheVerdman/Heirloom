use crate::shape::checked_numel;
use crate::{DType, Device, Result, Tensor, TensorError};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 6] = b"\x93NUMPY";

pub fn write_npy(path: impl AsRef<Path>, tensor: &Tensor) -> Result<()> {
    if tensor.device() != Device::Cpu {
        return Err(TensorError::InvalidOperation(format!(
            "write_npy supports only CPU tensors, got {:?}",
            tensor.device(),
        )));
    }
    if tensor.dtype() == DType::BFloat16 {
        return Err(TensorError::DType(
            "write_npy does not support Heirloom BFloat16 tensors yet".to_string(),
        ));
    }

    let mut file = File::create(path.as_ref()).map_err(|err| {
        TensorError::Io(format!(
            "failed to create {}: {err}",
            path.as_ref().display()
        ))
    })?;
    file.write_all(MAGIC)
        .map_err(|err| TensorError::Io(format!("failed to write npy magic: {err}")))?;
    file.write_all(&[1, 0])
        .map_err(|err| TensorError::Io(format!("failed to write npy version: {err}")))?;

    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr_for_dtype(tensor.dtype()),
        shape_tuple(&tensor.shape())
    );
    let preamble_len = MAGIC.len() + 2 + 2;
    let padding = (16 - ((preamble_len + header.len() + 1) % 16)) % 16;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    if header.len() > u16::MAX as usize {
        return Err(TensorError::Io(format!(
            "npy v1 header too large: {} bytes",
            header.len()
        )));
    }
    file.write_all(&(header.len() as u16).to_le_bytes())
        .map_err(|err| TensorError::Io(format!("failed to write npy header length: {err}")))?;
    file.write_all(header.as_bytes())
        .map_err(|err| TensorError::Io(format!("failed to write npy header: {err}")))?;

    match tensor.dtype() {
        DType::F32 => {
            for value in tensor.data_f32()? {
                file.write_all(&value.to_le_bytes()).map_err(|err| {
                    TensorError::Io(format!("failed to write npy f32 tensor data: {err}"))
                })?;
            }
        }
        DType::BFloat16 => unreachable!("BFloat16 rejected before writing npy header"),
        DType::F64 => {
            for value in tensor.data_f64_exact()? {
                file.write_all(&value.to_le_bytes()).map_err(|err| {
                    TensorError::Io(format!("failed to write npy f64 tensor data: {err}"))
                })?;
            }
        }
        DType::I64 => {
            for value in tensor.data_i64()? {
                file.write_all(&value.to_le_bytes()).map_err(|err| {
                    TensorError::Io(format!("failed to write npy i64 tensor data: {err}"))
                })?;
            }
        }
        DType::Bool => {
            for value in tensor.data_bool()? {
                file.write_all(&[u8::from(value)]).map_err(|err| {
                    TensorError::Io(format!("failed to write npy bool tensor data: {err}"))
                })?;
            }
        }
    }
    Ok(())
}

pub fn read_npy(path: impl AsRef<Path>, requires_grad: bool) -> Result<Tensor> {
    let mut file = File::open(path.as_ref()).map_err(|err| {
        TensorError::Io(format!("failed to open {}: {err}", path.as_ref().display()))
    })?;
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|err| TensorError::Io(format!("failed to read npy magic: {err}")))?;
    if &magic != MAGIC {
        return Err(TensorError::Io("invalid npy magic".to_string()));
    }

    let mut version = [0u8; 2];
    file.read_exact(&mut version)
        .map_err(|err| TensorError::Io(format!("failed to read npy version: {err}")))?;
    let header_len = match version {
        [1, 0] => {
            let mut len = [0u8; 2];
            file.read_exact(&mut len).map_err(|err| {
                TensorError::Io(format!("failed to read npy v1 header length: {err}"))
            })?;
            u16::from_le_bytes(len) as usize
        }
        [2, 0] | [3, 0] => {
            let mut len = [0u8; 4];
            file.read_exact(&mut len).map_err(|err| {
                TensorError::Io(format!("failed to read npy v2/v3 header length: {err}"))
            })?;
            u32::from_le_bytes(len) as usize
        }
        other => {
            return Err(TensorError::Io(format!(
                "unsupported npy version {}.{}",
                other[0], other[1]
            )))
        }
    };

    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|err| TensorError::Io(format!("failed to read npy header: {err}")))?;
    let header = std::str::from_utf8(&header)
        .map_err(|err| TensorError::Io(format!("npy header is not utf8/ascii: {err}")))?;
    let (dtype, shape) = parse_header(header)?;
    let len = checked_numel(&shape).map_err(|error| {
        TensorError::Io(format!("npy shape has an invalid element count: {error}"))
    })?;
    match dtype {
        DType::F32 => {
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                let mut bytes = [0u8; 4];
                file.read_exact(&mut bytes).map_err(|err| {
                    TensorError::Io(format!("failed to read npy f32 data: {err}"))
                })?;
                data.push(f32::from_le_bytes(bytes));
            }
            Tensor::from_f32(data, &shape, requires_grad)
        }
        DType::BFloat16 => Err(TensorError::DType(
            "read_npy does not support BFloat16 tensors in this prototype".to_string(),
        )),
        DType::F64 => {
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                let mut bytes = [0u8; 8];
                file.read_exact(&mut bytes).map_err(|err| {
                    TensorError::Io(format!("failed to read npy f64 data: {err}"))
                })?;
                data.push(f64::from_le_bytes(bytes));
            }
            Tensor::from_f64(data, &shape, requires_grad)
        }
        DType::I64 => {
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                let mut bytes = [0u8; 8];
                file.read_exact(&mut bytes).map_err(|err| {
                    TensorError::Io(format!("failed to read npy i64 data: {err}"))
                })?;
                data.push(i64::from_le_bytes(bytes));
            }
            Tensor::from_i64(data, &shape, requires_grad)
        }
        DType::Bool => {
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                let mut byte = [0u8; 1];
                file.read_exact(&mut byte).map_err(|err| {
                    TensorError::Io(format!("failed to read npy bool data: {err}"))
                })?;
                data.push(byte[0] != 0);
            }
            Tensor::from_bool(data, &shape, requires_grad)
        }
    }
}

fn parse_header(header: &str) -> Result<(DType, Vec<usize>)> {
    let descr = parse_quoted_field(header, "descr")?;
    let dtype = dtype_from_descr(descr)?;
    if !header.contains("'fortran_order': False")
        && !header.contains("\"fortran_order\": False")
        && !header.contains("\"fortran_order\": false")
    {
        return Err(TensorError::InvalidOperation(
            "read_npy supports only C-order arrays".to_string(),
        ));
    }
    Ok((dtype, parse_shape(header)?))
}

fn descr_for_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "<f4",
        DType::BFloat16 => "|V2",
        DType::F64 => "<f8",
        DType::I64 => "<i8",
        DType::Bool => "|b1",
    }
}

fn dtype_from_descr(descr: &str) -> Result<DType> {
    match descr {
        "<f4" | "|f4" => Ok(DType::F32),
        "<f8" | "|f8" => Ok(DType::F64),
        "<i8" => Ok(DType::I64),
        "|b1" | "?" => Ok(DType::Bool),
        _ => Err(TensorError::DType(format!(
            "read_npy supports little-endian f32/f64/i64 and bool, got descr {descr:?}"
        ))),
    }
}

fn parse_quoted_field<'a>(header: &'a str, field: &str) -> Result<&'a str> {
    let field_pos = header
        .find(field)
        .ok_or_else(|| TensorError::Io(format!("missing npy header field {field:?}")))?;
    let after_field = &header[field_pos + field.len()..];
    let colon = after_field
        .find(':')
        .ok_or_else(|| TensorError::Io(format!("missing ':' after field {field:?}")))?;
    let after_colon = after_field[colon + 1..].trim_start();
    let quote = after_colon
        .chars()
        .next()
        .ok_or_else(|| TensorError::Io(format!("empty value for npy field {field:?}")))?;
    if quote != '\'' && quote != '"' {
        return Err(TensorError::Io(format!(
            "expected quoted value for npy field {field:?}"
        )));
    }
    let rest = &after_colon[quote.len_utf8()..];
    let end = rest
        .find(quote)
        .ok_or_else(|| TensorError::Io(format!("unterminated quoted npy field {field:?}")))?;
    Ok(&rest[..end])
}

fn parse_shape(header: &str) -> Result<Vec<usize>> {
    let shape_pos = header
        .find("shape")
        .ok_or_else(|| TensorError::Io("missing npy shape field".to_string()))?;
    let after_shape = &header[shape_pos..];
    let open = after_shape
        .find('(')
        .ok_or_else(|| TensorError::Io("missing '(' in npy shape".to_string()))?;
    let after_open = &after_shape[open + 1..];
    let close = after_open
        .find(')')
        .ok_or_else(|| TensorError::Io("missing ')' in npy shape".to_string()))?;
    let body = after_open[..close].trim();
    if body.is_empty() {
        return Ok(Vec::new());
    }

    let mut dims = Vec::new();
    for part in body.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        dims.push(part.parse::<usize>().map_err(|err| {
            TensorError::Shape(format!("invalid npy shape dimension {part:?}: {err}"))
        })?);
    }
    Ok(dims)
}

fn shape_tuple(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [only] => format!("({},)", only),
        _ => {
            let body = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({body})")
        }
    }
}
