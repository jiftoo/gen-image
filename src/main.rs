use std::fs;
use std::{fmt::Debug, path::Path};

use std::{
	collections::hash_map::{DefaultHasher, RandomState},
	hash::{BuildHasher, Hasher},
};

use glium::{glutin, implement_vertex, Surface};
use rand::prelude::Distribution;

// #[derive(Debug, Clone, Copy)]
// enum ImageType {
// 	Png,
// 	Jpeg,
// }

// #[derive(Clone)]
// struct OpenImage {
// 	path: PathBuf,
// 	image: ColorImage,
// }

// impl OpenImage {
// 	fn load<T: AsRef<Path>>(path: T) -> Result<Self, String> {
// 		let image = {
// 			let file_data = fs::read(path.as_ref()).map_err(|x| x.to_string())?;
// 			Self::read_image(&file_data)
// 		};

// 		Ok(Self {
// 			path: path.as_ref().to_owned(),
// 			image,
// 		})
// 	}

// 	fn read_image(data: &[u8]) -> ColorImage {
// 		let image = image::io::Reader::new(Cursor::new(data))
// 			.with_guessed_format()
// 			.unwrap()
// 			.decode()
// 			.unwrap();

// 		let size = [image.width() as usize, image.height() as usize];

// 		let rgba8 = image
// 			.into_rgba8()
// 			.pixels()
// 			.map(|x| Color32::from_rgba_unmultiplied(x.0[0], x.0[1], x.0[2], x.0[3]))
// 			.collect();

// 		ColorImage {
// 			pixels: rgba8,
// 			size,
// 		}
// 	}

// 	fn create_texture(&self, ctx: &egui::Context) -> TextureHandle {
// 		let image_data = ImageData::Color(self.image.clone());
// 		ctx.load_texture("canvas", image_data, Default::default())
// 	}
// }

// struct App {
// 	open_file: Option<OpenImage>,
// 	open_msg: String,
// 	image_texture: Option<TextureHandle>,
// }

// impl App {
// 	fn new() -> Self {
// 		Self {
// 			open_file: None,
// 			open_msg: "".into(),
// 			image_texture: None,
// 		}
// 	}
// }

// impl eframe::App for App {
// 	fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
// 		egui::CentralPanel::default().show(ctx, |ui| {
// 			ui.vertical_centered(|ui| {
// 				if ui.button("Select image...").clicked() {
// 					if let Some(path) = rfd::FileDialog::new()
// 						.add_filter("image", &["png", "jpg", "jpeg"])
// 						.pick_file()
// 					{
// 						let (open_msg, open_file) = OpenImage::load(path).map_or_else(
// 							|x| (x, None),
// 							|x| (format!("Opened: {}", x.path.display()), Some(x)),
// 						);
// 						self.open_file = open_file;
// 						self.open_msg = open_msg;
// 						if let Some(open_file) = &self.open_file {
// 							self.image_texture = Some(open_file.create_texture(ctx));
// 						}
// 					}
// 				}
// 				ui.label(&self.open_msg);
// 			});

// 			if let Some(image) = &self.image_texture {
// 				let mn = (ui.available_size().min_elem() - 32.0).max(0.0);
// 				let i_w = image.size()[0] as f32;
// 				let i_h = image.size()[1] as f32;
// 				let i_r = image.aspect_ratio();
// 				let s_r = 1.0;
// 				let size = if s_r > i_r {
// 					(mn * i_r, mn)
// 				} else {
// 					(mn, i_h * mn / i_w)
// 				};
// 				ui.vertical_centered(|ui| {
// 					ui.image(image, size);
// 				});
// 			}
// 		});
// 	}
// }

// fn main() {
// 	eframe::run_native(
// 		"gen-image",
// 		eframe::NativeOptions {
// 			default_theme: eframe::Theme::Light,
// 			initial_window_size: Some(Vec2::new(500.0, 500.0)),
// 			..Default::default()
// 		},
// 		Box::new(|_| Box::new(App::new())),
// 	)
// 	.unwrap();
// }

// fn gen_vertices(n_triangles: usize) -> Vec<Vertex> {
// 	let mut rng = rand::thread_rng();
// 	let dist = rand::distributions::Uniform::new(0.0f32, 1.0f32);

// 	let mut uninit_vertex = (0..(n_triangles * 3))
// 		.map(|_| {
// 			let (x, y) = (
// 				rand::thread_rng().gen_range(-1.0f32..=1.0f32),
// 				rand::thread_rng().gen_range(-1.0f32..=1.0f32),
// 			);
// 			let mut vx = MaybeUninit::<Vertex>::uninit();
// 			unsafe {
// 				addr_of_mut!((*vx.as_mut_ptr()).position).write([x, y]);
// 			}
// 			vx
// 		})
// 		.collect::<Vec<_>>();

// 	uninit_vertex
// 		.chunks_mut(3)
// 		.flat_map(|ch| {
// 			let [v1, v2, v3]: &mut [_; 3] = ch.try_into().unwrap();
// 			let color = [
// 				dist.sample(&mut rng),
// 				dist.sample(&mut rng),
// 				dist.sample(&mut rng),
// 				dist.sample(&mut rng),
// 			];
// 			unsafe {
// 				addr_of_mut!((*v1.as_mut_ptr()).color_rgba).write(color);
// 				addr_of_mut!((*v2.as_mut_ptr()).color_rgba).write(color);
// 				addr_of_mut!((*v3.as_mut_ptr()).color_rgba).write(color);

// 				[v1.assume_init(), v2.assume_init(), v3.assume_init()]
// 			}
// 		})
// 		.collect::<Vec<_>>()
// }

#[derive(Clone, Copy, Debug)]
struct Vertex {
	position: [f32; 2],
	color_rgba: [u8; 4],
}
implement_vertex!(Vertex, position, color_rgba);

struct FastRand {
	hasher: DefaultHasher,
}

impl FastRand {
	fn new() -> Self {
		Self {
			hasher: RandomState::new().build_hasher(),
		}
	}
	#[inline(always)]
	fn rand(&mut self) -> u64 {
		let val = self.hasher.finish();
		self.hasher.write_u64(val);
		val
	}

	fn randnormfloat(&mut self) -> f32 {
		(self.rand() % 100000) as f32 / 100000.0
	}
}

fn gen_vertices(n_triangles: usize) -> Vec<Vertex> {
	let mut h = FastRand::new();

	let mut vertices = Vec::with_capacity(n_triangles * 3);

	let mut color = [0; 4];
	for i in 0..(n_triangles * 3) {
		let x = ((h.rand() % 200000) as i64 - 100000) as f32 / 100000.0;
		let y = ((h.rand() % 200000) as i64 - 100000) as f32 / 100000.0;
		if i % 3 == 0 {
			color = [
				(h.rand() % 256) as u8,
				(h.rand() % 256) as u8,
				(h.rand() % 256) as u8,
				(h.rand() % 256) as u8,
			];
		}

		let vertex = Vertex {
			position: [x, y],
			color_rgba: color,
		};
		vertices.push(vertex);
	}

	vertices
}

// #[repr(C)]
// #[derive(Clone, Copy, Debug)]
// struct DiffComputeOutput {
// 	sq_diff: [f32; 256 * 256],
// }
// implement_uniform_block!(DiffComputeOutput, sq_diff);

struct LossComputeProgram {
	compute_program: glium::program::ComputeShader,
	diff_texture: glium::texture::Texture2d,
	// uniform_buffer: glium::uniforms::UniformBuffer<[u32; 256*256]>,
}

impl LossComputeProgram {
	fn build(display: &glium::Display, width: u32, height: u32) -> Self {
		let compute_shader_src = r#"
		#version 440

		layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

		// layout(std140) buffer Output {
		// 	uint sq_diff[];
		// };

		layout(r32f) writeonly uniform image2D diff_texture;

		layout(rgba8) readonly uniform image2D current_image;
		layout(rgba8) readonly uniform image2D reference_image;

		void main() {
			// why does imageLoad accept ivec and not uvec?

            vec3 rgba = imageLoad(current_image, ivec2(gl_GlobalInvocationID.xy)).rgb;
            vec3 ref_rgba = imageLoad(reference_image, ivec2(gl_GlobalInvocationID.xy)).rgb;
            vec3 diff = rgba - ref_rgba;

            imageStore(diff_texture, ivec2(gl_GlobalInvocationID.xy), vec4(dot(diff, diff)));
		}
	"#;

		// let uniform_buffer =
		// 	glium::uniforms::UniformBuffer::<[u32; 256*256]>::empty_dynamic(display).unwrap();

		let diff_texture = glium::texture::Texture2d::empty_with_format(
			display,
			glium::texture::UncompressedFloatFormat::U8U8U8U8,
			glium::texture::MipmapsOption::NoMipmap,
			width,
			height,
		)
		.unwrap();

		let compute_program =
			glium::program::ComputeShader::from_source(display, compute_shader_src).unwrap();

		LossComputeProgram {
			compute_program,
			diff_texture, // uniform_buffer,
		}
	}

	#[inline]
	fn compute_loss(&self, current: &glium::Texture2d, reference: &glium::Texture2d) -> f32 {
		let w = current.width();
		let h = current.height();

		assert!(w == reference.width());
		assert!(h == reference.height());

		// self.uniform_buffer.invalidate();
		let uniforms = glium::uniform! {
			// Output: &self.uniform_buffer,
			diff_texture: self.diff_texture.image_unit(glium::uniforms::ImageUnitFormat::R32F).unwrap(),
			current_image: current.image_unit(glium::uniforms::ImageUnitFormat::RGBA8).unwrap(),
			reference_image: reference.image_unit(glium::uniforms::ImageUnitFormat::RGBA8).unwrap(),
		};

		self.compute_program.execute(uniforms, w, h, 1);

		let pixels: Vec<Vec<(u8, u8, u8, u8)>> = self.diff_texture.read();
		pixels
			.into_iter()
			.flatten()
			.map(|x| f32::from_ne_bytes([x.0, x.1, x.2, x.3]))
			.sum::<f32>()
	}
}

struct PolygonRenderer<'a> {
	display: &'a glium::Display,
	width: u32,
	height: u32,
	reference_image: &'a glium::texture::Texture2d,
	vbo: glium::VertexBuffer<Vertex>,
	draw_program: glium::program::Program,
	compute_program: LossComputeProgram,
}

impl<'a> PolygonRenderer<'a> {
	fn build(
		display: &'a glium::Display,
		width: u32,
		height: u32,
		polygons: &[Vertex],
		reference_image: &'a glium::texture::Texture2d,
	) -> Self {
		let vertex_shader_src = r#"
		#version 430

		in vec2 position;
		in uvec4 color_rgba;

		out vec4 color_in;

		void main() {
			color_in = vec4(color_rgba) / 255;
			gl_Position = vec4(position, 0.0, 1.0);
		}
	"#;
		let fragment_shader_src = r#"
		#version 430

		in vec4 color_in;

		out vec4 color_out;

		void main() {
			color_out = color_in;
		}
	"#;

		let program =
			glium::Program::from_source(display, vertex_shader_src, fragment_shader_src, None)
				.unwrap();

		// let mut texture_framebuffer =
		// 	glium::framebuffer::SimpleFrameBuffer::new(display, &texture).unwrap();

		PolygonRenderer {
			display,
			width,
			height,
			reference_image,
			vbo: glium::VertexBuffer::dynamic(display, polygons).unwrap(),
			draw_program: program,
			compute_program: LossComputeProgram::build(display, width, height),
		}
	}

	fn gen_texture(&self) -> glium::Texture2d {
		glium::texture::Texture2d::empty_with_format(
			self.display,
			glium::texture::UncompressedFloatFormat::U8U8U8U8,
			glium::texture::MipmapsOption::NoMipmap,
			self.width,
			self.height,
		)
		.unwrap()
	}

	#[inline]
	fn update_polygon(&mut self, base: usize, polygons: &[Vertex]) {
		let slice = self.vbo.slice(base..(base + 3)).unwrap();
		slice.invalidate();
		slice.write(polygons);
	}

	#[inline]
	fn clear_draw_polygons(&self, target: &mut glium::framebuffer::SimpleFrameBuffer) {
		target.clear_color(0.6, 0.6, 0.6, 1.0);
		target
			.draw(
				&self.vbo,
				glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
				&self.draw_program,
				&glium::uniforms::EmptyUniforms,
				&glium::DrawParameters {
					blend: glium::Blend::alpha_blending(),
					..Default::default()
				},
			)
			.unwrap();
	}

	#[inline]
	fn calculate_loss(&self, current: &glium::texture::Texture2d) -> f32 {
		self.compute_program
			.compute_loss(current, self.reference_image)
	}
}

fn build_image_buffer(
	src: &glium::texture::Texture2d,
) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
	let pixels: Vec<Vec<(u8, u8, u8, u8)>> = src.read();

	let mut image_buffer = image::ImageBuffer::new(pixels[0].len() as u32, pixels.len() as u32);
	for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
		let (r, g, b, a) = pixels[y as usize][x as usize];
		*pixel = image::Rgba([r, g, b, a]);
	}

	image_buffer
}

struct Population {
	members: Vec<Vertex>,
	mutation_backup: Option<(usize, [Vertex; 3])>,
	rand: FastRand,
}

impl Population {
	fn random(size: usize) -> Self {
		Population {
			members: gen_vertices(size),
			mutation_backup: None,
			rand: FastRand::new(),
		}
	}

	#[inline]
	fn mutate_color(&mut self, base: usize) {
		let v123 = &mut self.members[base..(base + 3)];

		let rnd = self.rand.rand();
		let rnd_a = (self.rand.rand() % 256) as u8;

		v123[0].color_rgba[(rnd % 3) as usize] = rnd_a;
		v123[1].color_rgba[(rnd % 3) as usize] = rnd_a;
		v123[2].color_rgba[(rnd % 3) as usize] = rnd_a;
	}

	#[inline]
	fn mutate_coord(&mut self, base: usize) {
		let v123 = &mut self.members[base..(base + 3)];

		let rnd = self.rand.rand();
		let rnd_a = self.rand.rand();
		let rnd_f = self.rand.randnormfloat();

		v123[(rnd % 3) as usize].position[(rnd_a % 2) as usize] = rnd_f * 2.0 - 1.0;
	}

	#[inline]
	fn mutate_alpha(&mut self, base: usize) {
		let v123 = &mut self.members[base..(base + 3)];
		let rnd_a = (self.rand.rand() % 256) as u8;

		v123[0].color_rgba[3] = rnd_a;
		v123[1].color_rgba[3] = rnd_a;
		v123[2].color_rgba[3] = rnd_a;
	}

	#[inline]
	fn mutate(&mut self) -> (&[Vertex], usize) {
		let base = self.rand.rand() as usize % (self.members.len() / 3) * 3;

		let v123 = &self.members[base..(base + 3)];
		self.mutation_backup = Some((base, [v123[0], v123[1], v123[2]]));

		let rnd = self.rand.rand() % 16;

		if rnd <= 2 {
			self.mutate_color(base);
		} else if rnd <= 8 {
			self.mutate_coord(base);
		} else {
			self.mutate_alpha(base);
		}

		(&self.members[base..(base + 3)], base)
	}

	#[inline]
	fn restore(&mut self) -> (&[Vertex], usize) {
		if let Some(backup) = self.mutation_backup.take() {
			self.members[backup.0..(backup.0 + 3)].copy_from_slice(&backup.1);
			self.mutation_backup = None;

			(&self.members[backup.0..(backup.0 + 3)], backup.0)
		} else {
			unreachable!("restore called without backup");
		}
	}
}

fn load_file_to_texutre(
	display: &glium::Display,
	path: impl AsRef<Path>,
) -> glium::texture::Texture2d {
	let image = image::open(path).unwrap().to_rgba8();
	let image_dimensions = image.dimensions();
	let image =
		glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
	glium::texture::Texture2d::with_format(
		display,
		image,
		glium::texture::UncompressedFloatFormat::U8U8U8U8,
		glium::texture::MipmapsOption::NoMipmap,
	)
	.unwrap()
}

fn texture_to_scale(
	display: &glium::Display,
	src: &glium::Texture2d,
	dest_width: u32,
	dest_height: u32,
	srgb: bool,
) -> glium::Texture2d {
	let passthrough_compute = r#"
		#version 440

		layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

		uniform sampler2D tex;
        uniform bool srgb;

		layout(rgba8) writeonly uniform image2D outImage;

		void main() {
			uint x = gl_GlobalInvocationID.x;
			uint y = gl_GlobalInvocationID.y;
			ivec2 size = textureSize(tex, 0);

			vec4 color = texture(tex, vec2(x, y) / gl_NumWorkGroups.xy);

            // if (srgb) {
            //     color.rgb = pow(color.rgb, vec3(1.0 / 2.2));
            // }

			imageStore(outImage, ivec2(x, y), color);
		}
	"#;

	let program = glium::program::ComputeShader::from_source(display, passthrough_compute).unwrap();

	let dest_texture = glium::texture::Texture2d::empty_with_format(
		display,
		glium::texture::UncompressedFloatFormat::U8U8U8U8,
		glium::texture::MipmapsOption::NoMipmap,
		dest_width,
		dest_height,
	)
	.unwrap();

	let dest_texture_unit = dest_texture
		.image_unit(glium::uniforms::ImageUnitFormat::RGBA8)
		.unwrap();

	let t1 = std::time::Instant::now();
	program.execute(
		glium::uniform! {
			tex: src.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
			outImage: dest_texture_unit,
			srgb: srgb,
		},
		dest_width,
		dest_height,
		1,
	);
	let t2 = std::time::Instant::now();
	println!("time: {:?}", t2 - t1);

	dest_texture
}

struct IdentityDrawer {
	program: glium::program::Program,
	vbo: glium::VertexBuffer<Vertex>,
	inner_size: (f32, f32),
}

impl IdentityDrawer {
	fn new(display: &glium::Display) -> Self {
		let vertex_shader_src = r#"
            #version 430

            in vec2 position;
            in vec4 color_rgba;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
	    "#;
		let fragment_shader_src = r#"
            #version 430

            out vec4 color_out;

            uniform sampler2D tex;
            uniform vec2 inner_size;

            void main() {
                color_out = texture(tex, gl_FragCoord.xy / inner_size);
            }
	    "#;

		let program = glium::program::Program::from_source(
			display,
			vertex_shader_src,
			fragment_shader_src,
			None,
		)
		.unwrap();

		let vbo = glium::VertexBuffer::new(
			display,
			&[
				Vertex {
					position: [-1.0, -1.0],
					color_rgba: [0, 0, 0, 0],
				},
				Vertex {
					position: [1.0, -1.0],
					color_rgba: [0, 0, 0, 0],
				},
				Vertex {
					position: [1.0, 1.0],
					color_rgba: [0, 0, 0, 0],
				},
				Vertex {
					position: [-1.0, -1.0],
					color_rgba: [0, 0, 0, 0],
				},
				Vertex {
					position: [1.0, 1.0],
					color_rgba: [0, 0, 0, 0],
				},
				Vertex {
					position: [-1.0, 1.0],
					color_rgba: [0, 0, 0, 0],
				},
			],
		)
		.unwrap();

		let size = display.gl_window().window().inner_size();

		Self {
			program,
			vbo,
			inner_size: (size.width as f32, size.height as f32),
		}
	}

	fn draw_texture(&self, frame: &mut glium::Frame, tex: &glium::Texture2d) {
		frame
			.draw(
				&self.vbo,
				glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
				&self.program,
				&glium::uniform! {
					tex: tex.sampled(),
					inner_size: self.inner_size
				},
				&glium::DrawParameters {
					blend: glium::Blend::alpha_blending(),
					..Default::default()
				},
			)
			.unwrap();
	}
}

fn main() {
	const WIDTH: u32 = 500;
	const HEIGHT: u32 = 500;

	let event_loop = glutin::event_loop::EventLoop::new();
	let wb = glutin::window::WindowBuilder::new()
		.with_visible(false)
		.with_inner_size(glutin::dpi::PhysicalSize::new(WIDTH, HEIGHT));
	let cb = glutin::ContextBuilder::new();
	let display = glium::Display::new(wb, cb, &event_loop).unwrap();

	let reference_image = load_file_to_texutre(&display, "input.png");
	let reference_image = texture_to_scale(&display, &reference_image, WIDTH, HEIGHT, false);

	let mut population = Population::random(512);

	let mut renderer = PolygonRenderer::build(
		&display,
		WIDTH,
		HEIGHT,
		&population.members,
		&reference_image,
	);
	let texture = renderer.gen_texture();
	let mut framebuffer = texture.as_surface();

	let identity_drawer = IdentityDrawer::new(&display);
    // let gray_texture = texture_to_scale(&display, &load_file_to_texutre(&display, "gray.png"), WIDTH, HEIGHT, false);

	renderer.clear_draw_polygons(&mut framebuffer);

	let initial_loss = renderer.calculate_loss(&texture);
	let mut current_loss = renderer.calculate_loss(&texture);
	let mut good_mutations = 0;
	let mut bad_mutations = 0;

	let initial_timestamp = std::time::Instant::now();
	let mut timestamp = initial_timestamp;
	let loss_cutoff_percentage = 0.92;
	// let mut loss_history = Vec::new();
	for i in 0.. {
		let (mutated_vertices, base) = population.mutate();

		renderer.update_polygon(base, mutated_vertices);
		renderer.clear_draw_polygons(&mut framebuffer);

		let loss = renderer.calculate_loss(&texture);

		if loss > current_loss {
			let (restored_vertices, base) = population.restore();
			renderer.update_polygon(base, restored_vertices);
			bad_mutations += 1;
		} else {
			current_loss = loss;
			good_mutations += 1;

			let t2 = std::time::Instant::now();
			if (t2 - timestamp).as_secs_f32() > 1.75 {
				// loss_history.push((
				// 	(t2 - initial_timestamp).as_secs_f32(),
				// 	// (initial_loss - current_loss) / initial_loss,
				// 	current_loss,
				// ));

				let image_buffer = build_image_buffer(&texture);
				image::DynamicImage::ImageRgba8(image_buffer)
					.flipv()
					.save(format!("output/gen_{i}.png"))
					.unwrap();

				timestamp = t2;
				println!(
					"gen: {i}, loss: {current_loss}, similarity: {:.1}%",
					((initial_loss - current_loss) / initial_loss) * 100.0,
				);
			}

			if ((initial_loss - current_loss) / initial_loss) >= loss_cutoff_percentage {
				break;
			}

			// let mut frame = display.draw();
			// frame.clear_color(1.0, 1.0, 1.0, 1.0);
			// // identity_drawer.draw_texture(&mut frame, &reference_image);
			// identity_drawer.draw_texture(&mut frame, &texture);
			// frame.finish().unwrap();

			// display
			// 	.gl_window()
			// 	.window()
			// 	.set_title(&format!("gen: {i}, loss: {current_loss}"));
		}
	}

	// fs::write(
	// 	"loss_history.csv",
	// 	loss_history
	// 		.into_iter()
	// 		.fold("TimeX,LossLogFY\n".to_string(), |mut acc, (x, y)| {
	// 			acc.push_str(&format!("{},{}\n", x.log2(), y.log2()));
	// 			acc
	// 		}),
	// )
	// .unwrap();

	let time_taken = initial_timestamp.elapsed();
	let mut time_taken_millis = time_taken.as_millis();
	let ms = time_taken_millis % 1000;
	time_taken_millis /= 1000;
	let s = time_taken_millis % 60;
	time_taken_millis /= 60;
	let m = time_taken_millis % 60;
	let h = (time_taken_millis - m) / 60;

	println!("initial loss: {initial_loss}, loss: {current_loss}, good: {good_mutations}, bad: {bad_mutations}, time take: {h}:{m}:{s}.{ms}");
}
