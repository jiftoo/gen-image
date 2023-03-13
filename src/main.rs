use std::{
	borrow::Cow,
	fmt::Debug,
	fs,
	io::Cursor,
	path::{Path, PathBuf},
};

use egui::{Color32, ColorImage, ImageData, Pos2, Rect, Stroke, TextureHandle, Vec2};
use std::{
	collections::hash_map::{DefaultHasher, RandomState},
	hash::{BuildHasher, Hasher},
	mem::MaybeUninit,
	ptr::addr_of_mut,
	time::Duration,
};

use glium::{
	framebuffer, glutin, implement_uniform_block, implement_vertex, texture::UnsignedTexture2d,
	Surface,
};
use rand::{prelude::Distribution, Rng};

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
	color_rgba: [f32; 4],
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

	let mut color = [0.0f32; 4];
	for i in 0..(n_triangles * 3) {
		let x = ((h.rand() % 200000) as i64 - 100000) as f32 / 100000.0;
		let y = ((h.rand() % 200000) as i64 - 100000) as f32 / 100000.0;
		if i % 3 == 0 {
			color = [
				// todo: move to randnormfloat()
				(h.rand() % 100000) as f32 / 100000.0,
				(h.rand() % 100000) as f32 / 100000.0,
				(h.rand() % 100000) as f32 / 100000.0,
				(h.rand() % 100000) as f32 / 100000.0,
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DiffComputeOutput {
	sq_diff: f32,
}
implement_uniform_block!(DiffComputeOutput, sq_diff);

struct LossComputeProgram {
	compute_program: glium::program::ComputeShader,
	uniform_buffer: glium::uniforms::UniformBuffer<DiffComputeOutput>,
}

impl LossComputeProgram {
	fn build(display: &glium::Display) -> Self {
		let compute_shader_src = r#"
		#version 430

		layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

		layout(std140, location = 1) buffer Output {
			float sq_diff;
		};

		layout(rgba8) readonly uniform image2D current_image;
		layout(rgba8) readonly uniform image2D reference_image;

		void main() {
			// why does imageLoad accept ivec and not uvec?

			vec4 rgba = imageLoad(current_image, ivec2(gl_GlobalInvocationID.xy));
			vec4 ref_rgba = imageLoad(reference_image, ivec2(gl_GlobalInvocationID.xy));
			float diff = dot(ref_rgba.rgb - rgba.rgb, vec3(1, 1, 1));
			sq_diff += diff * diff;
		}
	"#;

		let uniform_buffer =
			glium::uniforms::UniformBuffer::<DiffComputeOutput>::empty_dynamic(display).unwrap();

		let compute_program =
			glium::program::ComputeShader::from_source(display, compute_shader_src).unwrap();

		LossComputeProgram {
			compute_program,
			uniform_buffer,
		}
	}

	fn compute_loss(&self, current: &glium::Texture2d, reference: &glium::Texture2d) -> f32 {
		let w = current.width();
		let h = current.height();

		assert!(w == reference.width());
		assert!(h == reference.height());

		self.uniform_buffer
			.write(&DiffComputeOutput { sq_diff: 0.0 });
		let uniforms = glium::uniform! {
			Output: &self.uniform_buffer,
			current_image: current.image_unit(glium::uniforms::ImageUnitFormat::RGBA8).unwrap(),
			reference_image: reference.image_unit(glium::uniforms::ImageUnitFormat::RGBA8).unwrap(),
		};

		self.compute_program.execute(uniforms, w, h, 1);

		self.uniform_buffer.read().unwrap().sq_diff
	}
}

struct PolygonRenderer<'a> {
	display: &'a glium::Display,
	width: u32,
	height: u32,
	reference_image: glium::texture::Texture2d,
	vbo: glium::VertexBuffer<Vertex>,
	vbo_size: usize,
	draw_program: glium::program::Program,
	compute_program: LossComputeProgram,
}

impl<'a> PolygonRenderer<'a> {
	fn build(
		display: &'a glium::Display,
		width: u32,
		height: u32,
		max_polygons: usize,
		reference_image: glium::texture::Texture2d,
	) -> Self {
		let vertex_shader_src = r#"
		#version 430

		in vec2 position;
		in vec4 color_rgba;

		out vec4 color_in;

		void main() {
			color_in = color_rgba;
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
			vbo_size: 0,
			vbo: glium::VertexBuffer::empty_dynamic(display, max_polygons * 3).unwrap(),
			draw_program: program,
			compute_program: LossComputeProgram::build(display),
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

	fn update_polygons(&mut self, polygons: &Vec<Vertex>) {
		self.vbo.invalidate();
		self.vbo.slice(0..polygons.len()).unwrap().write(polygons);
		self.vbo_size = polygons.len();
	}

	fn clear_draw_polygons(&self, target: &mut glium::framebuffer::SimpleFrameBuffer) {
		let slice = self.vbo.slice(0..self.vbo_size).unwrap();
		target.clear_color(0.0, 0.0, 0.0, 0.0);
		target
			.draw(
				slice,
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

	fn calculate_loss(&self, current: &glium::texture::Texture2d) -> f32 {
		self.compute_program
			.compute_loss(current, &self.reference_image)
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

enum MutationResult {
	Good,
	Bad,
}

struct Population {
	members: Vec<Vertex>,
	// mutation_backup: Option<Vec<Vertex>>,
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

	fn mutate(&mut self) -> &Vec<Vertex> {
		let base = self.rand.rand() as usize % (self.members.len() / 3) * 3;
		let v123: &mut [Vertex; 3] = &mut self.members[base..(base + 3)].try_into().unwrap();

		self.mutation_backup = Some((base, *v123));

		v123.copy_from_slice(&gen_vertices(1));

		// let [v1, v2, v3] = v123;

		// let judgement = self.rand.rand() % 10;
		// let mentjudge = (self.rand.rand() % 3) as usize;
		// let f = self.rand.randnormfloat();
		// if judgement <= 2 {
		// 	v1.color_rgba[mentjudge] = f;
		// 	v2.color_rgba[mentjudge] = f;
		// 	v3.color_rgba[mentjudge] = f;
		// } else {
		// 	let third = (self.rand.rand() % 2) as usize;
		// 	v123[mentjudge].position[third] = f * 2.0 - 1.0;
		// }

		&self.members
	}

	fn restore(&mut self) {
		if let Some(backup) = self.mutation_backup.take() {
			self.members[backup.0..(backup.0 + 3)].copy_from_slice(&backup.1);
			self.mutation_backup = None;
		} else {
			unreachable!("restore called without backup");
		}
	}

	// fn mutate(&mut self) -> &Vec<Vertex> {
	// 	for chunk in self.members.chunks_exact_mut(3) {
	// 		let rnd = self.rand.rand() % 10;
	// 		if rnd <= 3 {
	// 			let v = gen_vertices(1);
	// 			chunk[0] = v[0];
	// 			chunk[1] = v[1];
	// 			chunk[2] = v[2];
	// 		}
	// 	}

	// 	self.mutation_backup = Some(self.members.clone());

	// 	&self.members
	// }

	// fn restore(&mut self) {
	// 	if let Some(backup) = self.mutation_backup.take() {
	// 		self.members = backup;
	// 	} else {
	// 		unreachable!("restore called without backup");
	// 	}
	// }
}

fn prepare_gl_context(width: u32, height: u32) -> glium::Display {
	let event_loop = glutin::event_loop::EventLoop::new();
	let wb = glutin::window::WindowBuilder::new()
		.with_visible(false)
		.with_inner_size(glutin::dpi::PhysicalSize::new(width, height));
	let cb = glutin::ContextBuilder::new();
	glium::Display::new(wb, cb, &event_loop).unwrap()
}

fn load_file_to_texutre(
	display: &glium::Display,
	path: impl AsRef<Path>,
) -> glium::texture::Texture2d {
	let image = image::open(path).unwrap().to_rgba8();
	let image_dimensions = image.dimensions();
	let image = glium::texture::RawImage2d::from_raw_rgba(image.into_raw(), image_dimensions);
	glium::texture::Texture2d::new(display, image).unwrap()
}

fn texture_to_scale(
	display: &glium::Display,
	src: &glium::Texture2d,
	dest_width: u32,
	dest_height: u32,
) -> glium::Texture2d {
	let passthrough_compute = r#"
		#version 430

		layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

		uniform sampler2D tex;

		layout(rgba8) writeonly uniform image2D outImage;

		void main() {
			uint x = gl_GlobalInvocationID.x;
			uint y = gl_GlobalInvocationID.y;
			ivec2 size = textureSize(tex, 0);

			vec4 color = texture(tex, vec2(x, y) / gl_NumWorkGroups.xy);

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
			outImage: dest_texture_unit
		},
		dest_width,
		dest_height,
		1,
	);
	let t2 = std::time::Instant::now();
	println!("time: {:?}", t2 - t1);

	dest_texture
}

fn main() {
	const WIDTH: u32 = 256;
	const HEIGHT: u32 = 256;
	let display = prepare_gl_context(WIDTH, HEIGHT);

	let reference_image = load_file_to_texutre(&display, "input.png");
	let reference_image = texture_to_scale(&display, &reference_image, WIDTH, HEIGHT);

	let mut population = Population::random(256);

	let mut renderer = PolygonRenderer::build(&display, WIDTH, HEIGHT, 256, reference_image);
	let texture = renderer.gen_texture();
	let mut framebuffer = texture.as_surface();

	renderer.update_polygons(&population.members);
	renderer.clear_draw_polygons(&mut framebuffer);

	let mut current_loss = renderer.calculate_loss(&texture);
	for i in 0..100000 {
		let mutated_population = population.mutate();

		renderer.update_polygons(mutated_population);
		renderer.clear_draw_polygons(&mut framebuffer);

		let loss = renderer.calculate_loss(&texture);

		if loss > current_loss {
			population.restore();
		} else {
			current_loss = loss;
			let image_buffer = build_image_buffer(&texture);
			image_buffer.save(format!("test{i}.png")).unwrap();
			println!("loss: {}", loss);
		}
	}


	// let shape = gen_vertices(128);

	// println!("shape: {:?}", shape.len());

	// let texture = glium::texture::Texture2d::empty_with_format(
	// 	&display,
	// 	glium::texture::UncompressedFloatFormat::U8U8U8U8,
	// 	glium::texture::MipmapsOption::NoMipmap,
	// 	1024,
	// 	1024,
	// )
	// .unwrap();

	// let texture_unit = texture
	// 	.image_unit(glium::uniforms::ImageUnitFormat::RGBA8)
	// 	.unwrap()
	// 	.set_access(glium::uniforms::ImageUnitAccess::ReadWrite);

	// let program =
	// 	glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None)
	// 		.unwrap();

	// println!("initialized");

	// // for i in 0..1 {
	// {
	// 	let t1 = std::time::Instant::now();

	// 	let mut target = texture.as_surface(); //display.draw();
	// 	target.clear_color(0.0, 0.0, 0.0, 0.0);
	// 	target
	// 		.draw(
	// 			&vbo,
	// 			indices,
	// 			&program,
	// 			// &glium::uniforms::EmptyUniforms,
	// 			&glium::uniform! {
	// 				tex: texture_unit,
	// 			},
	// 			&glium::DrawParameters {
	// 				blend: glium::Blend::alpha_blending(),
	// 				..Default::default()
	// 			},
	// 		)
	// 		.unwrap();

	// 	// target.finish().unwrap();

	// 	// let target = display.draw();
	// 	// 	.as_surface()
	// 	// 	.fill(&target, glium::uniforms::MagnifySamplerFilter::Nearest);

	// 	// target.finish().unwrap();

	// 	let t2 = std::time::Instant::now();

	// 	let new_vertices = gen_vertices(128);
	// 	vbo.write(&new_vertices);

	// 	let t3 = std::time::Instant::now();

	// 	let pixels: Vec<Vec<(u8, u8, u8, u8)>> = texture.read(); //display.read_front_buffer().unwrap();

	// 	let mut image_buffer = image::ImageBuffer::new(pixels[0].len() as u32, pixels.len() as u32);
	// 	for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
	// 		let (r, g, b, a) = pixels[y as usize][x as usize];
	// 		*pixel = image::Rgba([r, g, b, a]);
	// 	}

	// 	// let a = image_buffer
	// 	// 	.enumerate_pixels()
	// 	// 	.filter(|(x, y, p)| p != &&image::Rgba::<u8>([0, 0, 0, 255]))
	// 	// 	.count();

	// 	// println!("pixels: {}", a);

	// 	let t4 = std::time::Instant::now();

	// 	image_buffer.save(format!("output/image{}.png", 1)).unwrap();

	// 	let t5 = std::time::Instant::now();

	// 	println!(
	// 		"gen image draw:{:?}, upd:{:?}, buf:{:?}, write:{:?}",
	// 		t2 - t1,
	// 		t3 - t2,
	// 		t4 - t3,
	// 		t5 - t4
	// 	);
	// }

	// println!("rendered");
}
