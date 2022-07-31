extern crate opencv;

use crate::mark_seam;
use opencv::core::{add_weighted, Mat, MatTrait, MatTraitManual, Scalar, Vec3b, CV_8U};
use opencv::highgui;
use opencv::imgcodecs;
use opencv::imgcodecs::IMREAD_UNCHANGED;
use opencv::imgproc::sobel;
use std::cmp::{max, min};
use std::fmt::Error;

fn generate_energies(input_matrix: &Vec<Vec<u8>>) -> Result<Vec<Vec<usize>>, Error> {
    //create the energy matrix
    let mut energy_map = vec![vec![0; input_matrix[0].len()]; input_matrix.len()];

    //set the first line manually because it has no pixels above
    for x in 0..input_matrix[0].len() {
        energy_map[0][x] = input_matrix[0][x] as usize;
    }

    //iterater over every row but the first
    for y in 1..(energy_map.len()) {
        for x in 0..(energy_map[0].len()) {
            //get the value for the current pixel
            let pixel_value = energy_map[y][x];
            let energy_val: i32;

            //handle the left row separately
            if x == 0 {
                //calculate the energy value by adding the minimum of the energies
                //above to the pixel value
                energy_val =
                    (pixel_value + min(energy_map[y - 1][x + 1], energy_map[y - 1][x])) as i32;
            } else if x == energy_map[0].len() - 1 {
                //handle the right row separately
                //calculate the energy value by adding the minimum of the energies
                //above to the pixel value
                energy_val =
                    (pixel_value + min(energy_map[y - 1][x - 1], energy_map[y - 1][x])) as i32;
            } else {
                //get minimum of three pixels above and add
                let min_tmp = min(energy_map[y - 1][x], energy_map[y - 1][x - 1]);

                //calculate the energy value by adding the minimum of the energies
                //above to the pixel value
                energy_val = (pixel_value + min(energy_map[y - 1][x + 1], min_tmp)) as i32;
            }

            //assign the new energy value to the energy map
            energy_map[y][x] = min(energy_val as usize, usize::MAX);
        }
    }

    Result::Ok(energy_map)
}

fn generate_seam(energy_map: &Vec<Vec<usize>>) -> Vec<i32> {
    let mut minimum_indices = vec![-1 as i32; energy_map.len()];

    let mut minimum_val = i32::MAX;
    let mut min_index: i32 = -1;

    //find the first minimum value manually
    for x in 0..energy_map[0].len() {
        //get the energy value in the last row manually.
        let energy_value = energy_map[energy_map.len() - 1][x];

        if minimum_val > energy_value as i32 {
            minimum_val = energy_value as i32;
            min_index = x as i32;
        }
    }
    minimum_indices[0] = min_index;

    //iterate over the rows starting with the last one
    for y in (0..energy_map.len() - 1).rev() {
        let mut minimum_val: i32 = i32::MAX;
        let mut min_index: i32 = -1;

        //iterate over the three pixels before, exactly at and after the min_index
        //but also don't iterate over the end
        for x in max(minimum_indices[energy_map.len() - y - 2] - 1, 0)
            ..min(
                minimum_indices[energy_map.len() - y - 2] + 2,
                (energy_map[0].len() - 1) as i32,
            )
        {
            println!("x: {}, y: {}, {}", x, y, energy_map[y as usize][x as usize]);

            if energy_map[y as usize][x as usize] < minimum_val as usize {
                minimum_val = energy_map[y as usize][x as usize] as i32;
                min_index = x as i32;
            }
        }

        minimum_indices[energy_map.len() - y - 1] = min_index as i32;
    }

    minimum_indices
}

fn cut_seam(image: &Vec<Vec<u8>>, seam: &Vec<i32>) -> Result<Vec<Vec<u8>>, Error> {
    let mut output: Vec<Vec<u8>> = vec![vec![0 as u8; image[0].len()]; image.len()];

    //copy everything but the pixels at the index
    for y in 0..output.len() {
        let mut x_offset = 0;

        for x in 0..output[0].len() {
            if x == seam[y] as usize {
                x_offset = 1;
            }
            output[y][x] = image[y][x + x_offset];
        }
    }

    Result::Ok(output)
}

fn apply_gradient(image: &Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>, Error> {
    let image_as_matrix: Mat = vector_to_matrix(&image).unwrap();

    let mut grad_x: Mat = Mat::new_rows_cols_with_default(
        image_as_matrix.rows(),
        image_as_matrix.cols(),
        image_as_matrix.depth().unwrap(),
        Scalar::from(0.0),
    )
    .unwrap();

    let mut grad_y: Mat = grad_x.clone();
    let mut out: Mat = grad_x.clone();

    let _ = sobel(
        &image_as_matrix,
        &mut grad_x,
        image_as_matrix.depth().unwrap(),
        1,
        0,
        3,
        1.0,
        0.0,
        0,
    );

    let _ = sobel(
        &image_as_matrix,
        &mut grad_y,
        image_as_matrix.depth().unwrap(),
        0,
        1,
        3,
        1.0,
        0.0,
        0,
    );

    let _ = add_weighted(
        &grad_x,
        0.5,
        &grad_y,
        0.5,
        0.0,
        &mut out,
        image_as_matrix.depth().unwrap(),
    );

    Result::Ok(matrix_to_vector(&out).unwrap())
}

fn to_grayscale(image: &Mat) -> Result<Mat, Error> {
    let mut gray: Mat =
        Mat::new_rows_cols_with_default(image.rows(), image.cols(), CV_8U, Scalar::from(0.0))
            .unwrap();

    for y in 0..image.rows() {
        for x in 0..image.cols() {
            let r = image.at_2d::<Vec3b>(y, x).unwrap()[0] as f32;
            let g = image.at_2d::<Vec3b>(y, x).unwrap()[1] as f32;
            let b = image.at_2d::<Vec3b>(y, x).unwrap()[2] as f32;

            *gray.at_2d_mut::<u8>(y, x).unwrap() = ((r + g + b) / 3.0).floor() as u8;
        }
    }

    Ok(gray)
}

fn matrix_to_vector(matrix: &Mat) -> Result<Vec<Vec<u8>>, Error> {
    let mut vec_out: Vec<Vec<u8>> =
        vec![vec![0 as u8; matrix.cols() as usize]; matrix.rows() as usize];

    for y in 0..matrix.rows() {
        for x in 0..matrix.cols() {
            vec_out[y as usize][x as usize] = *matrix.at_2d::<u8>(y, x).unwrap();
        }
    }

    Ok(vec_out)
}

fn vector_to_matrix(vector: &Vec<Vec<u8>>) -> Result<Mat, Error> {
    let mut matrix_out = Mat::new_rows_cols_with_default(
        vector.len() as i32,
        vector[0].len() as i32,
        CV_8U,
        Scalar::from(0.0),
    )
    .unwrap();

    for y in 0..vector.len() {
        for x in 0..vector[0].len() {
            *matrix_out.at_2d_mut::<u8>(y as i32, x as i32).unwrap() = vector[y][x];
        }
    }

    Ok(matrix_out)
}
fn cut_seam_from_matrix(image: &Mat, seam: &Vec<i32>) -> Result<Mat, Error> {
    let mut output = Mat::new_rows_cols_with_default(
        image.rows(),
        image.cols() - 1,
        image.typ().unwrap(),
        Scalar::from(0.0),
    )
    .unwrap();

    //copy everything but the pixels at the index
    for y in 0..image.rows() {
        let mut x_offset = 0;

        for x in 0..output.cols() {
            if x == seam[y as usize] {
                x_offset = 1;
            }
            *output.at_2d_mut::<Vec3b>(y, x).unwrap() = *image.at_2d(y, x + x_offset).unwrap();
        }
    }

    Result::Ok(output)
}

fn carve(image: &Mat, num: i32) -> Result<Mat, Error> {
    let mut image_updated_colored = image.clone();
    let mut image_gray = to_grayscale(&image).unwrap().to_vec_2d().unwrap();

    for i in 0..num {
        println!("cutting seam {}", i);

        let gradient_matrix = apply_gradient(&image_gray).unwrap();

        let energy_map = generate_energies(&gradient_matrix).unwrap();

        let seam = generate_seam(&energy_map);
        println!("{:?}", seam);

        image_updated_colored = mark_seam(&image_updated_colored, &seam).unwrap();

        image_gray = cut_seam(&image_gray, &seam).unwrap();
    }

    Ok(image_updated_colored)
}

pub fn main() {
    let path: String = String::from("/home/clemens/repositorys/seamcarving/picture.bmp");
    let image = imgcodecs::imread(&path, IMREAD_UNCHANGED).unwrap();

    let carved = carve(&image, 1).unwrap();

    highgui::imshow("filtered", &carved).unwrap();
    highgui::wait_key(0).unwrap();
}
