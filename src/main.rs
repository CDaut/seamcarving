extern crate opencv;

use opencv::highgui;
use opencv::imgproc::sobel;
use opencv::imgcodecs;
use opencv::core::{Mat, MatTrait, add_weighted, Scalar, CV_32S};
use opencv::imgcodecs::IMREAD_GRAYSCALE;
use std::cmp::{min, max};

pub fn generate_energies(input_matrix: &Mat) -> Mat {
    let mut energy_map = Mat::new_rows_cols_with_default(input_matrix.rows(),
                                                         input_matrix.cols(),
                                                         CV_32S,
                                                         Scalar::new(0.0,
                                                                     0.0,
                                                                     0.0,
                                                                     0.0))
        .unwrap();

    for y in 1..(energy_map.rows()) {
        for x in 0..(energy_map.cols()) {
            let pixel_value = *input_matrix.at_2d::<u8>(y as i32, x as i32)
                .unwrap() as i32;

            //handle the left row separately
            if x == 0 {
                *energy_map.at_2d_mut::<i32>(y, x).unwrap() = pixel_value
                    + min(energy_map.at_2d::<i32>(y - 1, x + 1).unwrap(),
                          energy_map.at_2d::<i32>(y - 1, x).unwrap());
            } else if x == energy_map.cols() - 1 {
                //handle the right row separately
                *energy_map.at_2d_mut::<i32>(y, x).unwrap() = pixel_value
                    + min(energy_map.at_2d::<i32>(y - 1, x - 1).unwrap(),
                          energy_map.at_2d::<i32>(y - 1, x).unwrap());
            } else {
                //get minimum of three pixels above and add
                *energy_map.at_2d_mut::<i32>(y, x).unwrap() = pixel_value
                    + min(energy_map.at_2d::<i32>(y - 1, x + 1).unwrap(),
                          min(energy_map.at_2d::<i32>(y - 1, x).unwrap(),
                              energy_map.at_2d::<i32>(y - 1, x - 1).unwrap()));
            }
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
        let energy_value = *energy_map.at_2d::<i32>(energy_map.rows() - 1, x)
            .unwrap() as i32;

        if minimum_val > energy_value {
            minimum_val = energy_value;
            min_index = x as i32;
        }
    }
    minimum_indices.push(min_index);


    for y in (0..energy_map.rows() - 1).rev()
    {
        let mut minimum_val = i32::MAX;
        let mut min_index = -1;

        //iterate over the three pixels before, exactly at and after the min_index
        //but also don't iterate over the end
        for x in max(minimum_indices.last().unwrap() - 1, 0)..
            min(minimum_indices.last().unwrap() + 1, energy_map.cols() - 1) {
            if *energy_map.at_2d::<i32>(y, x).unwrap() < minimum_val {
                minimum_val = *energy_map.at_2d::<i32>(y, x).unwrap() as i32;
                min_index = x;
            }
        }
        minimum_indices.push(min_index);
    }

    minimum_indices
}

fn update_image(mut image: &mut Mat, seam: Vec<i32>) {
    for index in (0..seam.len()).rev() {
        *image.at_2d_mut::<u8>(index as i32,
                           *seam.get(seam.len() - (index + 1)).unwrap()).unwrap()
            = 0;
    }
}

fn main()
{
    let path: String = String::from("/home/clemens/repositorys/seamcarving/picture.bmp");

    let mut image = imgcodecs::imread(&path, IMREAD_GRAYSCALE)
        .unwrap(); // Issue over here


    let mut grad_x: Mat = Mat::new_rows_cols_with_default(image.rows(),
                                                          image.cols(),
                                                          image.depth().unwrap(),
                                                          Scalar::new(0.0,
                                                                      0.0,
                                                                      0.0,
                                                                      0.0))
        .unwrap();

    let mut grad_y: Mat = grad_x.clone();
    let mut out: Mat = grad_x.clone();

    let _ = sobel(&image, &mut grad_x,
                  image.depth().unwrap(),
                  1,
                  0,
                  3,
                  1.0,
                  0.0,
                  0);

    let _ = sobel(&image, &mut grad_y,
                  image.depth().unwrap(),
                  0,
                  1,
                  3,
                  1.0,
                  0.0,
                  0);

    let _ = add_weighted(&grad_x,
                         0.5,
                         &grad_y,
                         0.5,
                         0.0,
                         &mut out,
                         image.depth().unwrap());


    let energy_map: Mat = generate_energies(&out);

    let seam = generate_seam(&energy_map);

    update_image(&mut image, seam);

    highgui::imshow("filtered", &image).unwrap();
    highgui::wait_key(0).unwrap();
}