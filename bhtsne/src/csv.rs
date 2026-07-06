use std::{
    error::Error,
    fs::File,
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::Float;

use crate::{FittedBhtsne, FittedExact, FittedFitSne};

/// Writes a row-major embedding to a csv file. If `D` is 2 or 3 the file gets simple headers
/// (`x, y` or `x, y, z`); higher dimensionalities are written without headers.
fn write_embedding_csv<T, const D: usize>(embedding: &[T], path: &str) -> Result<(), Box<dyn Error>>
where
    T: Float + ToString,
{
    let mut writer = ::csv::Writer::from_path(path)?;
    let to_write: Vec<String> = embedding.iter().map(|&el| el.to_string()).collect();
    match D {
        2 => writer.write_record(["x", "y"])?,
        3 => writer.write_record(["x", "y", "z"])?,
        _ => (),
    }
    for record in to_write.chunks(D) {
        writer.write_record(record)?;
    }
    writer.flush()?;
    Ok(())
}

impl<'d, T, U, const D: usize> FittedBhtsne<'d, T, U, D>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
    U: Send + Sync,
{
    /// Writes the fitted embedding to a csv file. Adds `x, y` or `x, y, z` headers for `D` in
    /// `2..=3`, no headers otherwise.
    pub fn write_csv(&self, path: &str) -> Result<&Self, Box<dyn Error>>
    where
        T: Float + ToString,
    {
        write_embedding_csv::<T, D>(self.embedding(), path)?;
        Ok(self)
    }
}

impl<'d, T, U, const D: usize> FittedFitSne<'d, T, U, D>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
    U: Send + Sync,
{
    /// Writes the fitted embedding to a csv file, matching [`FittedBhtsne::write_csv`].
    pub fn write_csv(&self, path: &str) -> Result<&Self, Box<dyn Error>>
    where
        T: Float + ToString,
    {
        write_embedding_csv::<T, D>(self.embedding(), path)?;
        Ok(self)
    }
}

impl<'d, T, U, const D: usize> FittedExact<'d, T, U, D>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
    U: Send + Sync,
{
    /// Writes the fitted embedding to a csv file, matching [`FittedBhtsne::write_csv`].
    pub fn write_csv(&self, path: &str) -> Result<&Self, Box<dyn Error>>
    where
        T: Float + ToString,
    {
        write_embedding_csv::<T, D>(self.embedding(), path)?;
        Ok(self)
    }
}

/// Loads data from a csv file.
///
/// # Arguments
///
/// * `file_path` - path of the file to load the data from.
///
/// * `has_headers` - whether the file has headers or not. if set to `true` the function will
///   not parse the first line of the csv file.
///
/// * `skip` - an optional slice that specifies a subset of the file columns that must not be
///   parsed.
///
/// * `f` - function that converts [`String`] into a data sample. It takes as an argument a single
///   record field.
///
/// # Errors
///
/// Returns an error is something goes wrong during the I/O operations.
pub fn load_csv<T, F>(
    path: &str,
    has_headers: bool,
    skip: Option<&[usize]>,
    f: F,
) -> Result<Vec<T>, Box<dyn Error>>
where
    F: Fn(String) -> T,
{
    let mut data: Vec<T> = Vec::new();

    let file = File::open(path)?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(file);

    match skip {
        Some(range) => {
            for result in reader.records() {
                let record = result?;

                (0..record.len())
                    .filter(|column| !range.contains(column))
                    .for_each(|field| data.push(f(record.get(field).unwrap().to_string())));
            }
        }
        None => {
            for result in reader.records() {
                let record = result?;

                (0..record.len())
                    .for_each(|field| data.push(f(record.get(field).unwrap().to_string())));
            }
        }
    }

    Ok(data)
}
