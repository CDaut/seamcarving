mod with_vectors;

extern crate opencv;

use opencv::core::{add_weighted, Mat, MatTrait, Scalar, Vec3b, CV_32S, CV_8U};
use opencv::highgui;
use opencv::imgcodecs;
use opencv::imgcodecs::IMREAD_UNCHANGED;
use opencv::imgproc::sobel;
use std::cmp::{max, min};
use std::fmt::Error;

fn generate_energies(input_matrix: &Mat) -> Result<Mat, Error> {
    //create the energy matrix
    let mut energy_map = Mat::new_rows_cols_with_default(
        input_matrix.rows(),
        input_matrix.cols(),
        CV_32S,
        Scalar::from(0.0),
    )
    .unwrap();

    //set the first line manually because it has no pixels above
    for x in 0..input_matrix.cols() {
        *energy_map.at_2d_mut::<i32>(0, x).unwrap() = *energy_map.at_2d_mut::<i32>(0, x).unwrap();
    }

    //iterater over every row but the first
    for y in 1..(energy_map.rows()) {
        for x in 0..(energy_map.cols()) {
            //get the value for the current pixel
            let pixel_value = *input_matrix.at_2d::<u8>(y, x).unwrap() as i32;
            let energy_val: i32;

            //handle the left row separately
            if x == 0 {
                //calculate the energy value by adding the minimum of the energies
                //above to the pixel value
                energy_val = pixel_value
                    + *min(
                        energy_map.at_2d::<i32>(y - 1, x + 1).unwrap(),
                        energy_map.at_2d::<i32>(y - 1, x).unwrap(),
                    );
            } else if x == energy_map.cols() - 1 {
                //handle the right row separately
                //calculate the energy value by adding the minimum of the energies
                //above to the pixel value
                energy_val = pixel_value
                    + *min(
                        energy_map.at_2d::<i32>(y - 1, x - 1).unwrap(),
                        energy_map.at_2d::<i32>(y - 1, x).unwrap(),
                    );
            } else {
                //get minimum of three pixels above and add
                let min_tmp = min(
                    energy_map.at_2d::<i32>(y - 1, x).unwrap(),
                    energy_map.at_2d::<i32>(y - 1, x - 1).unwrap(),
                );

                //calculate the energy value by adding the minimum of the energies
                //above to the pixel value
                energy_val =
                    pixel_value + *min(energy_map.at_2d::<i32>(y - 1, x + 1).unwrap(), min_tmp);
            }

            //assign the new energy value to the energy map
            *energy_map.at_2d_mut::<i32>(y, x).unwrap() = min(energy_val, i32::MAX);
        }
    }

    Result::Ok(energy_map)
}

fn generate_seam(energy_map: &Mat) -> Vec<i32> {
    let mut minimum_indices = vec![];

    let mut minimum_val = i32::MAX;
    let mut min_index: i32 = -1;

    //find the first minimum value manually
    for x in 0..energy_map.cols() {
        //get the energy value in the last row manually.
        let energy_value = *energy_map.at_2d::<i32>(energy_map.rows() - 1, x).unwrap();

        if minimum_val > energy_value {
            minimum_val = energy_value;
            min_index = x;
        }
    }
    minimum_indices.push(min_index);

    for y in (0..energy_map.rows() - 1).rev() {
        let mut minimum_val = i32::MAX;
        let mut min_index = -1;

        //iterate over the three pixels before, exactly at and after the min_index
        //but also don't iterate over the end
        for x in max(minimum_indices.last().unwrap() - 1, 0)
            ..min(minimum_indices.last().unwrap() + 2, energy_map.cols() - 1)
        {
            if *energy_map.at_2d::<i32>(y, x).unwrap() < minimum_val {
                minimum_val = *energy_map.at_2d::<i32>(y, x).unwrap();
                min_index = x;
            }
        }
        minimum_indices.push(min_index);
    }

    minimum_indices
}

fn cut_seam(image: &Mat, seam: &Vec<i32>, colordepth: i32) -> Result<Mat, Error> {
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
            if colordepth == CV_32S {
                *output.at_2d_mut::<Vec3b>(y, x).unwrap() = *image.at_2d(y, x + x_offset).unwrap();
            } else if colordepth == CV_8U {
                *output.at_2d_mut::<u8>(y, x).unwrap() = *image.at_2d(y, x + x_offset).unwrap();
            }
        }
    }

    Result::Ok(output)
}

#[allow(dead_code)]
pub fn mark_seam(image: &Mat, seam: &Vec<i32>) -> Result<Mat, Error> {
    let mut output = image.clone();

    for index in (0..seam.len()).rev() {
        *output
            .at_2d_mut::<Vec3b>(index as i32, *seam.get(seam.len() - (index + 1)).unwrap())
            .unwrap() = Vec3b::from([0, 0, 255]);
    }

    Ok(output)
}

fn apply_gradient(image: &Mat) -> Result<Mat, Error> {
    let mut grad_x: Mat = Mat::new_rows_cols_with_default(
        image.rows(),
        image.cols(),
        image.depth().unwrap(),
        Scalar::from(0.0),
    )
    .unwrap();

    let mut grad_y: Mat = grad_x.clone();
    let mut out: Mat = grad_x.clone();

    let _ = sobel(
        &image,
        &mut grad_x,
        image.depth().unwrap(),
        1,
        0,
        3,
        1.0,
        0.0,
        0,
    );

    let _ = sobel(
        &image,
        &mut grad_y,
        image.depth().unwrap(),
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
        image.depth().unwrap(),
    );

    Result::Ok(out)
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

fn carve(image: &Mat, num: i32) -> Result<Mat, Error> {
    let mut image_updated_colored = image.clone();
    let mut image_gray = to_grayscale(&image).unwrap();

    for i in 0..num {
        println!("cutting seam {}", i);

        let gradient_matrix: Mat = apply_gradient(&image_gray).unwrap();
        let energy_map: Mat = generate_energies(&gradient_matrix).unwrap();
        let seam = generate_seam(&energy_map);

        image_updated_colored = cut_seam(&image_updated_colored, &seam, CV_32S).unwrap();

        image_gray = cut_seam(&image_gray, &seam, CV_8U).unwrap();
    }

    Ok(image_updated_colored)
}

fn main() {
    // let path: String = String::from("/home/clemens/repositorys/seamcarving/surfer.png");
    // let image = imgcodecs::imread(&path, IMREAD_UNCHANGED).unwrap();
    //
    // let carved = carve(&image, 70).unwrap();
    //
    // highgui::imshow("filtered", &carved).unwrap();
    // highgui::wait_key(0).unwrap();

    with_vectors::main();
}
