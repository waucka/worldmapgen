use clap::{Arg, App, SubCommand};
use simdnoise::*;
use nalgebra::base::Vector2;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use worldgenlib::{HeightMap, gen_wrapped_noise, NoiseParams, OctaveParams, Bands, ErosionConfig, ConeConfig, VolcanoConfig, ChunkConfig, CraterConfig, MountainConfig, MountainType};

use std::f32::consts::PI;

fn main() {
    let matches = App::new("worldmapgen")
        .version("1.0")
        .about("Generates a heightmap for a planet")
        .subcommand(
            SubCommand::with_name("noisegen")
                .about("generates noise")
                .arg(Arg::with_name("output")
                     .short("o")
                     .long("output")
                     .value_name("FILE")
                     .help("Specifies the file to write the resulting image to")
                     .required(true)
                     .takes_value(true))
                .arg(Arg::with_name("script")
                     .short("s")
                     .long("script")
                     .value_name("FILE")
                     .help("Specifies the path to a Lua script that describes the noise")
                     .required(true)
                     .takes_value(true))
        )
        .subcommand(
            SubCommand::with_name("terraingen")
                .about("generates planet heightmap")
                .arg(Arg::with_name("output")
                     .short("o")
                     .long("output")
                     .value_name("FILE")
                     .help("Specifies the file to write the resulting image to")
                     .required(true)
                     .takes_value(true))
        )
        .subcommand(
            SubCommand::with_name("mountaingen")
                .about("generates heightmap for a mountain")
                .arg(Arg::with_name("output")
                     .short("o")
                     .long("output")
                     .value_name("FILE")
                     .help("Specifies the file to write the resulting image to")
                     .required(true)
                     .takes_value(true))
                .arg(Arg::with_name("config")
                     .short("c")
                     .long("config")
                     .value_name("FILE")
                     .help("TOML-format config file for the mountain you want")
                     .required(true)
                     .takes_value(true))
                .arg(Arg::with_name("seed")
                     .long("seed")
                     .value_name("seed")
                     .help("Seed value for RNG (must be a valid i32, defaults to a time-based seed)")
                     .required(false)
                     .takes_value(true))
        )
        .subcommand(
            SubCommand::with_name("cratergen")
                .about("generates heightmap for a crater")
                .arg(Arg::with_name("output")
                     .short("o")
                     .long("output")
                     .value_name("FILE")
                     .help("Specifies the file to write the resulting image to")
                     .required(true)
                     .takes_value(true))
                .arg(Arg::with_name("config")
                     .short("c")
                     .long("config")
                     .value_name("FILE")
                     .help("TOML-format config file for the crater you want")
                     .required(true)
                     .takes_value(true))
                .arg(Arg::with_name("seed")
                     .long("seed")
                     .value_name("seed")
                     .help("Seed value for RNG (must be a valid i32, defaults to a time-based seed)")
                     .required(false)
                     .takes_value(true))
        ).get_matches();

    if let Some(matches) = matches.subcommand_matches("noisegen") {
        let t_start = std::time::Instant::now();
        let output_filename = matches.value_of("output").unwrap();
        let script_filename = matches.value_of("script").unwrap();
        let script = std::fs::read_to_string(script_filename).unwrap();
        noisegen(output_filename, &script);
        let elapsed = t_start.elapsed().as_millis();
        println!("Noise generation took {}ms", elapsed);
    } else if let Some(matches) = matches.subcommand_matches("terraingen") {
        let t_start = std::time::Instant::now();
        terraingen(matches.value_of("output").unwrap());
        let elapsed = t_start.elapsed().as_millis();
        println!("Terrain generation took {}ms", elapsed);
    } else if let Some(matches) = matches.subcommand_matches("mountaingen") {
        let rand_seed = match matches.value_of("seed") {
            Some(val) => val.parse().unwrap(),
            None => {
                use std::time::SystemTime;
                let time_since_epoch = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
                let nanos = time_since_epoch.subsec_nanos();
                i32::from_le_bytes(nanos.to_le_bytes())
            },
        };
        let config_file_path = matches.value_of("config").unwrap();
        let config_bytes = std::fs::read_to_string(config_file_path).unwrap();
        let config = toml::from_str(&config_bytes).unwrap();
        mountaingen(&config, rand_seed, matches.value_of("output").unwrap());
    } else if let Some(matches) = matches.subcommand_matches("cratergen") {
        let rand_seed = match matches.value_of("seed") {
            Some(val) => val.parse().unwrap(),
            None => {
                use std::time::SystemTime;
                let time_since_epoch = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
                let nanos = time_since_epoch.subsec_nanos();
                i32::from_le_bytes(nanos.to_le_bytes())
            },
        };
        let config_file_path = matches.value_of("config").unwrap();
        let config_bytes = std::fs::read_to_string(config_file_path).unwrap();
        let config = toml::from_str(&config_bytes).unwrap();
        cratergen(&config, rand_seed, matches.value_of("output").unwrap());
    } else {
        // I don't think this can actually happen; clap should prevent it.
        panic!("Invalid subcommand");
    }
}

mod lua_stuff {
    use mlua::prelude::*;
    use mlua::{UserData, UserDataMethods};

    #[derive(Copy, Clone)]
    pub struct OctaveParams(pub worldgenlib::OctaveParams);
    impl UserData for OctaveParams {
        fn add_methods<'lua, M: UserDataMethods<'lua, Self>>(methods: &mut M) {
        }
    }

    #[derive(Clone)]
    pub struct NoiseParams(pub worldgenlib::NoiseParams);
    impl UserData for NoiseParams {
        fn add_methods<'lua, M: UserDataMethods<'lua, Self>>(methods: &mut M) {
        }
    }

    #[derive(Clone)]
    pub struct HeightMap(pub worldgenlib::HeightMap);
    impl UserData for HeightMap {
        fn add_methods<'lua, M: UserDataMethods<'lua, Self>>(methods: &mut M) {
            methods.add_method_mut("normalize", |_, hm, ()| {
                hm.0.normalize();
                Ok(())
            });
            methods.add_method_mut("invert", |_, hm, ()| {
                hm.0.invert();
                Ok(())
            });
            methods.add_method_mut(
                "apply_bands",
                |_, hm, (num_bands, range_min, range_max, falloff): (usize, f32, f32, f32)| {
                    let bands = worldgenlib::Bands::new(num_bands, range_min, range_max, falloff);
                    hm.0.apply_bands(&bands);
                    Ok(())
                });
            methods.add_method_mut("gaussian_blur", |_, hm, radius: f32| {
                hm.0.gaussian_blur(radius);
                Ok(())
            });
            methods.add_method_mut("elevate", |_, hm, amount: f32| {
                hm.0.elevate(amount);
                Ok(())
            });
            methods.add_method_mut("constrain", |_, hm, (min, max): (f32, f32)| {
                hm.0.constrain(min, max);
                Ok(())
            });
            methods.add_method_mut("clamp", |_, hm, (min, max): (f32, f32)| {
                hm.0.clamp(min, max);
                Ok(())
            });
            methods.add_method_mut(
                "warp",
                |_, hm, (warp_x, warp_y): (HeightMap, HeightMap)| {
                    hm.0.warp(&warp_x.0, &warp_y.0);
                    Ok(())
                });
        }
    }

    fn make_octave_params(_lua: &Lua, (freq, amp): (f32, f32)) -> LuaResult<OctaveParams> {
        Ok(OctaveParams(worldgenlib::OctaveParams::new(freq, amp)))
    }

    fn make_noise_params(
        _lua: &Lua,
        (width, height, rand_seed, params): (u32, u32, i32, LuaTable)
    ) -> LuaResult<NoiseParams> {
        let params: Vec<worldgenlib::OctaveParams> = params.sequence_values()
            .map(|x: LuaResult<OctaveParams>| x.unwrap().0).collect();
        Ok(NoiseParams(worldgenlib::NoiseParams::new(width, height, rand_seed, &params)))
    }

    fn heightmap_from_noise(_lua: &Lua, (params,): (NoiseParams,)) -> LuaResult<HeightMap> {
        let params = &params.0;
        Ok(HeightMap(worldgenlib::HeightMap::from_noise(params)))
    }

    pub fn init(lua: &Lua) {
        let globals = lua.globals();

        let make_octave_params = lua.create_function(make_octave_params).unwrap();
        globals.set("make_octave_params", make_octave_params).unwrap();

        let make_noise_params = lua.create_function(make_noise_params).unwrap();
        globals.set("make_noise_params", make_noise_params).unwrap();

        let heightmap_from_noise = lua.create_function(heightmap_from_noise).unwrap();
        globals.set("heightmap_from_noise", heightmap_from_noise).unwrap();
    }
}

fn noisegen(output_filename: &str, script_text: &str) {
    use mlua::Lua;

    let lua = Lua::new();
    lua_stuff::init(&lua);
    let mut hm = None;
    lua.scope(|scope| {
        lua.globals().set(
            "return_heightmap",
            scope.create_function_mut(|_, heightmap: lua_stuff::HeightMap| {
                hm = Some(heightmap.0);
                Ok(())
            }).unwrap(),
        ).unwrap();
        lua.load(script_text).exec()
    }).unwrap();
    if let Some(hm) = hm {
        hm.save(output_filename).unwrap();
    }
}

fn terraingen(output_filename: &str) {
    //use rand::random;
    let rand_seed: u64 = 1;//random();
    let mut rng = SmallRng::seed_from_u64(rand_seed);
    // Width is twice 360 (2x horizontal resolution)
    // Height is twice 85 * 2 (2x vertical resolution excluding the poles)
    // These will be defined statically, since the noise we're working with is not terribly
    // high-res anyway, and generating it is expensive.
    // I tried 7200x3400, and it didn't seem any more detailed.
    let width: u32 = 360 * 2;
    let height: u32 = 85 * 2 * 2;
    let full_height: u32 = 90 * 2 * 2;
    let margin_size = (full_height - height) / 2;
    let mut terrain = {
        let mut noise = HeightMap::from_noise(&NoiseParams::from_pairs(
            width, full_height,
            rng.gen(),
            &[
                (1.0 * 10.0, 0.5),
                (2.0 * 10.0, 0.25),
                (4.0 * 10.0, 0.125),
                (8.0 * 10.0, 0.0625),
            ],
        ));

        noise.constrain(0.15, 0.55);
        noise.save("base_noise.png").unwrap();
        noise
    };

    let hill_country = {
        let warp_amount = 0.01;
        let multiplier = 80.0;
        let multiplier_warp = 240.0;

        let octaves = [
            OctaveParams::new(1.0 * multiplier, 1.0),
            OctaveParams::new(2.0 * multiplier, 0.5),
            OctaveParams::new(4.0 * multiplier, 0.25),
            OctaveParams::new(8.0 * multiplier, 1.125),
        ];

        let warp_octaves = [
            OctaveParams::new(1.0 * multiplier_warp, 1.0),
            OctaveParams::new(2.0 * multiplier_warp, 0.5),
            OctaveParams::new(4.0 * multiplier_warp, 0.25),
            OctaveParams::new(8.0 * multiplier_warp, 0.125),
        ];

        let params = NoiseParams::new(width, full_height, rng.gen(), &octaves);
        let warp_x_params = NoiseParams::new(width, full_height, rng.gen(), &warp_octaves);
        let warp_y_params = NoiseParams::new(width, full_height, rng.gen(), &warp_octaves);

        let mut noise = HeightMap::from_noise(&params);
        let mut warp_x_noise = HeightMap::from_noise(&warp_x_params);
        let mut warp_y_noise = HeightMap::from_noise(&warp_y_params);
        warp_x_noise.constrain(-warp_amount, warp_amount);
        warp_y_noise.constrain(-warp_amount, warp_amount);
        noise.warp(&warp_x_noise, &warp_y_noise);

        noise.normalize();
        //noise.elevate(0.3);
        //noise.clamp(0.0, 1.0);
        //noise.normalize();
        noise.constrain(0.5, 1.0);
        //noise.scale(10.0);
        noise.save("hill_country.png");
        noise
    };
    let mut mountains = HeightMap::new(width, full_height, 0.0);
    let mountain_choices = [
        (HeightMap::load("mountain_test.bin").unwrap(), 0.75),
        (HeightMap::load("mountain_test2.bin").unwrap(), 0.24),
        (HeightMap::load("mountain_test3.bin").unwrap(), 0.01),
    ];
    let num_mountain_choices = mountain_choices.len();
    for _ in 0..400 {
        let mountain = {
            let mountain_rand = rng.gen_range(0.0..1.0);
            let mut weight_sum = 0.0;
            let mut mountain = &mountain_choices[num_mountain_choices - 1].0;
            for (m, weight) in mountain_choices.iter() {
                if mountain_rand >= weight_sum && mountain_rand < weight_sum + *weight {
                    mountain = m;
                }
                weight_sum += *weight;
            }
            mountain
        };
        mountains.add_item(
            mountain,
            Vector2::new(
                rng.gen_range(0.0..720.0),
                rng.gen_range(0.0..360.0),
            ),
            64.0,
            rng.gen_range(0.7..1.5),
        );
    }
    mountains.constrain(0.0, 2.0);

    let mut mountains_copy = mountains.clone();
    mountains_copy.normalize();
    mountains_copy.save("mountains_noise.png").unwrap();

    let sea_regions = gen_wrapped_noise(&NoiseParams::from_pairs(
        width, full_height,
        rng.gen::<i32>() + 50,
        &[
            (1.0 * 20.0, 1.0),
            (2.0 * 20.0, 0.5),
            (4.0 * 20.0, 0.25),
            (8.0 * 20.0, 0.125),
        ],
    ));
    //let mut sea_regions = HeightMap::new(width, full_height, -5.0);

    let mountains_regions = gen_wrapped_noise(&NoiseParams::from_pairs(
        width, full_height,
        rng.gen(),
        &[
            (1.0 * 20.0, 1.0),
            (2.0 * 40.0, 0.5),
            (4.0 * 40.0, 0.25),
            (8.0 * 40.0, 0.125),
        ],
    ));

    let mut mountains_regions = HeightMap::from_flat(
        width,
        full_height,
        mountains_regions,
    );
    {
        let warp_amount = 0.05;
        let multiplier_warp = 480.0;
        let warp_octaves = [
            OctaveParams::new(1.0 * multiplier_warp, 1.0),
            OctaveParams::new(2.0 * multiplier_warp, 0.5),
            OctaveParams::new(4.0 * multiplier_warp, 0.25),
            OctaveParams::new(8.0 * multiplier_warp, 0.125),
        ];

        let warp_x_params = NoiseParams::new(width, full_height, rng.gen(), &warp_octaves);
        let warp_y_params = NoiseParams::new(width, full_height, rng.gen(), &warp_octaves);

        let mut warp_x_noise = HeightMap::from_noise(&warp_x_params);
        let mut warp_y_noise = HeightMap::from_noise(&warp_y_params);
        warp_x_noise.constrain(-warp_amount, warp_amount);
        warp_y_noise.constrain(-warp_amount, warp_amount);
        mountains_regions.warp(&warp_x_noise, &warp_y_noise);
    }
    mountains_regions.normalize();
    mountains_regions.elevate(0.1);
    mountains_regions.clamp(0.0, 1.0);
    for v in mountains_regions.iter_mut() {
        if *v > 0.75 {
            *v = 1.0;
        } else if *v < 0.25 {
            *v = 0.0;
        }
    }
    let bands = Bands::new(2, 0.0, 1.0, 1.0);
    mountains_regions.apply_bands(&bands);
    let regions_clear = mountains_regions.clone();
    mountains_regions.gaussian_blur(10.0);
    mountains_regions.save("mountains_regions_noise.png").unwrap();

    let hill_country_regions = gen_wrapped_noise(&NoiseParams::from_pairs(
        width, full_height,
        rng.gen(),
        &[
            (1.0 * 20.0, 1.0),
            (2.0 * 40.0, 0.5),
            (4.0 * 40.0, 0.25),
            (8.0 * 40.0, 0.125),
        ],
    ));

    let mut hill_country_regions = HeightMap::from_flat(
        width,
        full_height,
        hill_country_regions,
    );
    {
        let warp_amount = 0.05;
        let multiplier_warp = 480.0;
        let warp_octaves = [
            OctaveParams::new(1.0 * multiplier_warp, 1.0),
            OctaveParams::new(2.0 * multiplier_warp, 0.5),
            OctaveParams::new(4.0 * multiplier_warp, 0.25),
            OctaveParams::new(8.0 * multiplier_warp, 0.125),
        ];

        let warp_x_params = NoiseParams::new(width, full_height, rng.gen(), &warp_octaves);
        let warp_y_params = NoiseParams::new(width, full_height, rng.gen(), &warp_octaves);

        let mut warp_x_noise = HeightMap::from_noise(&warp_x_params);
        let mut warp_y_noise = HeightMap::from_noise(&warp_y_params);
        warp_x_noise.constrain(-warp_amount, warp_amount);
        warp_y_noise.constrain(-warp_amount, warp_amount);
        hill_country_regions.warp(&warp_x_noise, &warp_y_noise);
    }
    hill_country_regions.normalize();
    hill_country_regions.elevate(0.1);
    hill_country_regions.clamp(0.0, 1.0);
    for v in hill_country_regions.iter_mut() {
        if *v > 0.75 {
            *v = 1.0;
        } else if *v < 0.25 {
            *v = 0.0;
        }
    }
    let bands = Bands::new(2, 0.0, 1.0, 1.0);
    hill_country_regions.apply_bands(&bands);
    let regions_clear = hill_country_regions.clone();
    hill_country_regions.gaussian_blur(10.0);
    hill_country_regions.save("hill_country_regions_noise.png").unwrap();

    terrain.add_layer(&mountains_regions, &mountains);
    terrain.blend_layer(&hill_country_regions, &hill_country);

    let mut sea_regions = HeightMap::from_flat(
        width,
        full_height,
        sea_regions,
    );
    {
        let warp_amount = 0.05;
        let multiplier_warp = 480.0;
        let warp_octaves = [
            OctaveParams::new(1.0 * multiplier_warp, 1.0),
            OctaveParams::new(2.0 * multiplier_warp, 0.5),
            OctaveParams::new(4.0 * multiplier_warp, 0.25),
            OctaveParams::new(8.0 * multiplier_warp, 0.125),
        ];

        let warp_x_params = NoiseParams::new(width, full_height, rng.gen(), &warp_octaves);
        let warp_y_params = NoiseParams::new(width, full_height, rng.gen(), &warp_octaves);

        let mut warp_x_noise = HeightMap::from_noise(&warp_x_params);
        let mut warp_y_noise = HeightMap::from_noise(&warp_y_params);
        warp_x_noise.constrain(-warp_amount, warp_amount);
        warp_y_noise.constrain(-warp_amount, warp_amount);
        sea_regions.warp(&warp_x_noise, &warp_y_noise);
    }
    sea_regions.normalize();
    sea_regions.elevate(0.1);
    sea_regions.clamp(0.0, 1.0);
    let bands = Bands::new(2, 0.0, 1.0, 1.0);
    sea_regions.apply_bands(&bands);
    {
        let mut sea_regions_copy = sea_regions.clone();
        sea_regions_copy.normalize();
        sea_regions_copy.save("sea_regions_noise.png").unwrap();
    }
    sea_regions.constrain(-1.0, 0.0);
    let hm_zero = HeightMap::new(width, full_height, 0.0);
    sea_regions.blend_variable(&hm_zero, &regions_clear);
    sea_regions.gaussian_blur(5.0);
    {
        let mut sea_regions_copy = sea_regions.clone();
        sea_regions_copy.normalize();
        sea_regions_copy.save("sea_regions_noise_clipped.png").unwrap();
    }
    sea_regions.constrain(0.0, 1.0);
    let sea_bottom = HeightMap::new(width, full_height, -1.0);
    terrain.blend_variable(&sea_bottom, &sea_regions);

    let mut heightmap = terrain;

    {
        let mut hm = heightmap.clone();
        hm.normalize();
        hm.save("before_scaling.png");
    }

    let erosion_config = ErosionConfig{
        inertia: 0.025,
        sediment_capacity_factor: 4.0,
        min_sediment_capacity: 0.01,
        erode_speed: 0.5,
        deposit_speed: 0.1,
        evaporate_speed: 0.01,
        gravity: 4.0,
        max_droplet_lifetime: 30,
        num_droplets: 600000,
        erosion_radius: 2.0,
        initial_water: 1.0,
        initial_speed: 1.0,
    };
    {
        let mut hm = heightmap.clone();
        hm.normalize();
        hm.save("before_erode.png").unwrap();
    }
    let t_start = std::time::Instant::now();
    sea_regions.constrain(0.0, 1.0);
    let mask = sea_regions;//.scaled_dimensions(heightmap.width(), heightmap.height());
    //let mask = HeightMap::new(heightmap.width(), heightmap.height(), 1.0);
    //heightmap.erode(&erosion_config, 0, true, &mask);
    let elapsed = t_start.elapsed().as_millis();
    {
        let mut hm = heightmap.clone();
        hm.normalize();
        hm.save("eroded.png").unwrap();
    }
    println!("Erosion on a {}x{} map took {}ms", heightmap.width(), heightmap.height(), elapsed);

    let mut heightmap = heightmap.scaled_dimensions(7200, 3600);
    {
        let mut hm = heightmap.clone();
        hm.normalize();
        hm.save("rescaled.png");
    }
    let mut heightmap = heightmap.cropped(0, margin_size * 10, width * 10, height * 10);

    // Region around the poles that should be perfectly flat, in degrees
    let polar_exclusion_zone = 20.0;
    // Region around the polar exclusion zone that should smoothly transition to the zone's
    // height, in degrees
    let polar_transition_zone = 10.0;
    let mut polar_noise = gen_wrapped_noise(&NoiseParams::from_pairs(
        720, 1,
        rng.gen(),
        &[
            (1.0 * 80.0, 0.2),
            (2.0 * 80.0, 0.2 / 2.0),
            (4.0 * 80.0, 0.2 / 4.0),
            (8.0 * 80.0, 0.2 / 8.0),
        ],
    ));
    let polar_noise_resolution = 720.0;
    let mut polar_noise_avg = 0.0;
    for v in polar_noise.iter() {
        polar_noise_avg += *v;
    }
    polar_noise_avg /= polar_noise_resolution;
    for v in polar_noise.iter_mut() {
        *v = 1.0 + (*v - polar_noise_avg);
    }

    heightmap.constrain(0.0, 1.0);

    let mut north_pole_avg = 0.0;
    for x in 0..width {
        north_pole_avg += heightmap.get(x, 0);
    }
    north_pole_avg /= width as f32;

    let mut south_pole_avg = 0.0;
    for x in 0..width {
        south_pole_avg += heightmap.get(x, height - 1);
    }
    south_pole_avg /= width as f32;

    heightmap.save("gall.png").unwrap();
    let spherical = heightmap.to_spherical(
        20,
        polar_exclusion_zone,
        polar_transition_zone,
        rng.gen(),
    );
    spherical.save(output_filename).unwrap();
}

fn make_cone_mountain(size: u32, rand_seed: i32, config: &ConeConfig) -> (HeightMap, Vec<f32>) {
    let mut heightmap = HeightMap::new(size, size, 0.0);
    let half_size = (size / 2) as f32;
    let center_x = half_size;
    let center_y = half_size;
    // Noise for the starting cone base
    let base_noise = NoiseBuilder::fbm_1d(360)
        .with_freq(1.0 / (2.0 * PI))
        .with_seed(rand_seed)
        .generate_scaled(
            half_size * config.base_radius * (1.0 - config.base_variability),
            half_size * config.base_radius,
        );
    let peak_noise = NoiseBuilder::fbm_1d(360)
        .with_freq(1.0 / (2.0 * PI))
        .with_seed(rand_seed)
        .generate_scaled(
            half_size * config.peak_radius * (1.0 - config.peak_variability),
            half_size * config.peak_radius,
        );
    // Generate the starting heightmap: a truncated cone with a top and base of varying radius
    for x_int in 0..size {
        for y_int in 0..size {
            let x = x_int as f32 - center_x;
            let y = y_int as f32 - center_y;
            let theta = f32::atan2(y, x) + PI;
            let theta_idx = ((theta / (2.0 * PI)) * 360.0) as usize;
            let theta_idx = theta_idx.clamp(0, 359);
            let base_dist = base_noise[theta_idx];
            let peak_dist = peak_noise[theta_idx];
            let r = f32::sqrt(x * x + y * y);
            let v = heightmap.get_mut(x_int, y_int);
            if r > base_dist {
                *v = 0.0;
            } else if r < peak_dist {
                *v = 1.0;
            } else {
                let peak_to_base = base_dist - peak_dist;
                let ratio = 1.0 - (r - peak_dist) / peak_to_base;
                *v = ratio;
            }
        }
    }
    heightmap.gaussian_blur(2.0);
    (heightmap, base_noise)
}

fn make_volcano_mountain(size: u32, rand_seed: i32, config: &VolcanoConfig) -> (HeightMap, Vec<f32>) {
    let mut heightmap = HeightMap::new(size, size, 0.0);
    let half_size = (size / 2) as f32;
    let center_x = half_size;
    let center_y = half_size;
    let mut rng = SmallRng::seed_from_u64(rand_seed as u64);
    // Noise for the starting cone base
    let base_noise = NoiseBuilder::fbm_1d(360)
        .with_freq(0.02 * PI)
        .with_seed(rand_seed)
        .generate_scaled(0.6 * half_size, 0.9 * half_size);
    let caldera_outer_radius = rng.gen_range(0.1..0.4);
    let caldera_inner_radius = rng.gen_range((caldera_outer_radius * 0.1)..(caldera_outer_radius * 0.9));
    let caldera_height = rng.gen_range(0.2..0.7);
    let caldera_variability = 0.1;
    let caldera_outer_noise = NoiseBuilder::fbm_1d(360)
        .with_freq(0.02 * PI)
        .with_seed(rand_seed)
        .generate_scaled(
            caldera_outer_radius * half_size * (1.0 - caldera_variability),
            caldera_outer_radius * half_size);
    let caldera_inner_noise = NoiseBuilder::fbm_1d(360)
        .with_freq(0.02 * PI)
        .with_seed(rand_seed)
        .generate_scaled(
            caldera_inner_radius * half_size * (1.0 - caldera_variability),
            caldera_inner_radius * half_size);
    // Generate the starting heightmap: a truncated cone with a top and base of varying radius
    for x_int in 0..size {
        for y_int in 0..size {
            let x = x_int as f32 - center_x;
            let y = y_int as f32 - center_y;
            let theta = f32::atan2(y, x) + PI;
            let theta_idx = ((theta / (2.0 * PI)) * 360.0) as usize;
            let theta_idx = theta_idx.clamp(0, 359);
            let base_dist = base_noise[theta_idx];
            let caldera_outer_dist = caldera_outer_noise[theta_idx];
            let caldera_inner_dist = caldera_inner_noise[theta_idx];
            let r = f32::sqrt(x * x + y * y);
            let v = heightmap.get_mut(x_int, y_int);
            if r > base_dist {
                *v = 0.0;
            } else if r < caldera_inner_dist {
                *v = caldera_height;
            } else if r < caldera_outer_dist {
                let inner_to_outer = caldera_outer_dist - caldera_inner_dist;
                let rim_to_crater = 1.0 - caldera_height;
                let delta = r - caldera_inner_dist;
                let height = delta * rim_to_crater / inner_to_outer + caldera_height;
                *v = height;
            } else {
                let top_to_base = base_dist - caldera_outer_dist;
                let ratio = 1.0 - (r - caldera_outer_dist) / top_to_base;
                *v = ratio;
            }
        }
    }
    heightmap.gaussian_blur(2.0);
    (heightmap, base_noise)
}

fn make_chunk_mountain(size: u32, rand_seed: i32, config: &ChunkConfig) -> (HeightMap, Vec<f32>) {
    panic!("Not implemented!");
}

fn mountaingen(config: &MountainConfig, rand_seed: i32, output_filename: &str) {
    let size: u32 = 512;
    let half_size = (size / 2) as f32;
    let center_x = half_size;
    let center_y = half_size;
    let (mut heightmap, base_noise) = match &config.mountain {
        MountainType::Cone(type_config) => make_cone_mountain(size, rand_seed, type_config),
        MountainType::Volcano(type_config) => make_volcano_mountain(size, rand_seed, type_config),
        MountainType::Chunk(type_config) => make_chunk_mountain(size, rand_seed, type_config),
    };
    heightmap.noisify(config.noise_scale, rand_seed);
    if let Some(ref erosion_config) = config.erosion {
        let mask = HeightMap::new(size, size, 1.0);
        heightmap.erode(erosion_config, rand_seed as u64, false, &mask);
    }
    for x_int in 0..size {
        for y_int in 0..size {
            let x = x_int as f32 - center_x;
            let y = y_int as f32 - center_y;
            let theta = f32::atan2(y, x) + PI;
            let theta_idx = ((theta / (2.0 * PI)) * 360.0) as usize;
            let theta_idx = theta_idx.clamp(0, 359);
            let base_dist = base_noise[theta_idx];
            let r = f32::sqrt(x * x + y * y);
            let v = heightmap.get_mut(x_int, y_int);
            if r > base_dist {
                *v = 0.0;
            }
        }
    }
    heightmap.constrain(0.0, 1.0);
    heightmap.save(output_filename).unwrap();
}

// STFU, you obnoxious pedant.
#[allow(clippy::many_single_char_names)]
fn cratergen(config: &CraterConfig, rand_seed: i32, output_filename: &str) {
    let size: u32 = 512;
    let half_size = (size / 2) as f32;
    let center_x = half_size;
    let center_y = half_size;
    let mut crater = HeightMap::new(size, size, 0.0);
    let mut rng = SmallRng::seed_from_u64(rand_seed as u64);
    // Noise for the starting cone base
    let base_radius = config.base_radius;
    let base_variability = config.base_variability;
    let base_noise = NoiseBuilder::fbm_1d(360)
        .with_freq(1.0 / (2.0 * PI))
        .with_seed(rand_seed)
        .generate_scaled(
            base_radius * half_size * (1.0 - base_variability),
            base_radius * half_size,
        );
    let floor_height = config.fill_height;
    let depth = config.excavation_depth;
    let rim_steepness = rng.gen_range(0.1..1.0);
    let rim_radius = config.rim_radius;
    let rim_variability = config.rim_variability;
    let rim_noise = NoiseBuilder::fbm_1d(360)
        .with_freq(1.0 / (2.0 * PI))
        .with_seed(rand_seed)
        .generate_scaled(
            rim_radius * half_size * (1.0 - rim_variability),
            rim_radius * half_size,
        );
    for x_int in 0..size {
        for y_int in 0..size {
            let x = x_int as f32 - center_x;
            let y = y_int as f32 - center_y;
            let theta = f32::atan2(y, x) + PI;
            let theta_idx = ((theta / (2.0 * PI)) * 360.0) as usize;
            let theta_idx = theta_idx.clamp(0, 359);
            let base_dist = base_noise[theta_idx];
            let rim_dist = rim_noise[theta_idx];
            let r = f32::sqrt(x * x + y * y);
            let v = crater.get_mut(x_int, y_int);
            if r > base_dist {
                *v = 0.0;
            } else {
                let cavity_c = (rim_dist * rim_dist) / (depth + 1.0);
                let cavity_height = (r * r) / cavity_c - depth;
                let x = r - base_dist;
                let rim_c = rim_steepness / ((rim_dist - base_dist) * (rim_dist - base_dist));
                let rim_height = rim_c * x * x;
                *v = smooth_min(
                    smooth_min(cavity_height, rim_height, 0.5),
                    floor_height,
                    -0.5,
                );
            }
        }
    }
    crater.noisify(config.noise_scale, rand_seed);
    crater.gaussian_blur(2.0);
    if let Some(ref erosion_config) = config.erosion {
        let mask = HeightMap::new(size, size, 1.0);
        crater.erode(erosion_config, rand_seed as u64, false, &mask);
    }
    for x_int in 0..size {
        for y_int in 0..size {
            let x = x_int as f32 - center_x;
            let y = y_int as f32 - center_y;
            let theta = f32::atan2(y, x) + PI;
            let theta_idx = ((theta / (2.0 * PI)) * 360.0) as usize;
            let theta_idx = theta_idx.clamp(0, 359);
            let base_dist = base_noise[theta_idx];
            let r = f32::sqrt(x * x + y * y);
            let v = crater.get_mut(x_int, y_int);
            if r > base_dist {
                *v = 0.0;
            }
        }
    }
    if output_filename.ends_with(".bin") {
        crater.scale(1.0);
    } else {
        // Image formats generally don't like negative values.
        crater.normalize();
    }

    crater.save(output_filename).unwrap();
}

fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = f32::clamp((b - a + k) / (2.0 * k), 0.0, 1.0);
    a * h + b * (1.0 - h) - k * h * (1.0 - h)
}
