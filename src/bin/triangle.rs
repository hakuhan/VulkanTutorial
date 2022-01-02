use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::sync::Arc;

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> 
{
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images.iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

fn main()
{
    // Create instance and extensions of instance
    let required_extensions = vulkano_win::required_extensions();

    let instance = Instance::new(None, &required_extensions, None).unwrap();

    // Choose physical device
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    println!("Using device:{} (type: {:?})", physical.name(), physical.ty());

    // create window
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    // Create GPU queue
    let queue_family = physical
        .queue_families()
        .find(|&q|{
            // Get first queue
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap();

    // Initialize device
    let device_ext = DeviceExtensions{
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    // Get first element of queue
    let queue = queues.next().unwrap();

    // Create swapchain
    let (mut swapchain, images) = {
        // Querying the capabilities of surface
        let caps = surface.capabilities(physical).unwrap();

        // Alpha mode
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have
        let format = caps.supported_formats[0].0;

        // Demensions of window
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .unwrap()
    };

    // Create buffer
    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex 
        {
            position: [f32; 2],
        }
        vulkano::impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, [
            Vertex {
                position: [-0.5, -0.25],
            },
            Vertex {
                position: [0.0, 0.5],
            },
            Vertex {
                position: [0.25, -0.1],
            },
        ]
        .iter()
        .cloned()
        )
        .unwrap()
    };

    // Create shader
    mod vs 
    {
        vulkano_shaders::shader! 
        {
            ty: "vertex",
            src: "
                #version 450

                layout(location = 0) in vec2 position;

                void main()
                {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            "
        }
    }

    mod fs
    {
        vulkano_shaders::shader!
        {
            ty: "fragment",
            src: "
                #version 450

                layout(location = 0) out vec4 f_color;

                void main()
                {
                    f_color = vec4(0.5, 0.5, 1.0, 0.5);
                }
            "
        }
    }

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    // Create render pass
    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: 
            {
                color: 
                {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: 
            {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap(),
    );

    // Create pipeline
    let pipeline = Arc::new(
        GraphicsPipeline::start()  
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    // State of viewport
    let mut dynamic_state = DynamicState
    {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    // Create framebuffer
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| {
        match event
        {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow =  ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = 
                        match swapchain.recreate_with_dimensions(dimensions) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };
                    swapchain = new_swapchain;

                    framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state,);
                    recreate_swapchain = false;
                }

                // Check image
                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

                // Build command buffer
                let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                    .unwrap();
                
                builder.begin_render_pass(framebuffers[image_num].clone(), SubpassContents::Inline, clear_values,)
                    .unwrap()
                    .draw(
                        pipeline.clone(), 
                        &dynamic_state, 
                        vertex_buffer.clone(), (), (),)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(queue.clone(), command_buffer)
                        .unwrap()
                        .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                        .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _=> (),
        }
    });

}
