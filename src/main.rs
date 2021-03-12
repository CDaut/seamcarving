extern crate opencv;

use opencv::core::{add_weighted, Mat, MatTrait, Scalar, Vec3b, CV_32S};
use opencv::highgui;
use opencv::imgcodecs;
use opencv::imgcodecs::{IMREAD_GRAYSCALE, IMREAD_UNCHANGED};
use opencv::imgproc::sobel;
use std::cmp::{max, min};

fn generate_energies(input_matrix: &Mat) -> Mat {
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

    energy_map
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

fn update_image(image: &mut Mat, seam: Vec<i32>) {
    for index in (0..seam.len()).rev() {
        *image
            .at_2d_mut::<Vec3b>(index as i32, *seam.get(seam.len() - (index + 1)).unwrap())
            .unwrap() = Vec3b::from([0, 0, 255]);
    }
}

fn main() {
    let path: String = String::from("/home/clemens/repositorys/seamcarving/picture.bmp");

    let image_gray = imgcodecs::imread(&path, IMREAD_GRAYSCALE).unwrap();
    let mut image_colored = imgcodecs::imread(&path, IMREAD_UNCHANGED).unwrap();

    let mut grad_x: Mat = Mat::new_rows_cols_with_default(
        image_gray.rows(),
        image_gray.cols(),
        image_gray.depth().unwrap(),
        Scalar::from(0.0),
    )
    .unwrap();

    let mut grad_y: Mat = grad_x.clone();
    let mut out: Mat = grad_x.clone();

    let _ = sobel(
        &image_gray,
        &mut grad_x,
        image_gray.depth().unwrap(),
        1,
        0,
        3,
        1.0,
        0.0,
        0,
    );

    let _ = sobel(
        &image_gray,
        &mut grad_y,
        image_gray.depth().unwrap(),
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
        image_gray.depth().unwrap(),
    );

    let energy_map: Mat = generate_energies(&out);

    let seam = generate_seam(&energy_map);

    update_image(&mut image_colored, seam);

    highgui::imshow("filtered", &image_colored).unwrap();
    highgui::wait_key(0).unwrap();
}
