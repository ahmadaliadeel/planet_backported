use bevy::render::mesh::{Indices, Mesh};
//use bevy::render::pipeline::PrimitiveTopology;
use bevy::{
    prelude::*,
    render::{
        camera::Camera,
       // wireframe::{Wireframe, WireframeConfig, WireframePlugin},
    },
    utils::{Duration, Instant},
    //wgpu::{WgpuFeature, WgpuFeatures, WgpuOptions},
};

use bevy::window::Windows;
use futures_lite::future;
use noise::{NoiseFn, Fbm};//RidgedMulti as Fbm};
use wgpu::PrimitiveTopology;
use std::collections::hash_map::HashMap;
use bevy::tasks::{AsyncComputeTaskPool, Task};

fn f(p: Vec3) -> f32 {
    //0.0
    let mut fbm = Fbm::new();
    //fbm.octaves = 4;
    let p = p * 0.01f32;
    10f32 * fbm.get([p.x as f64, p.y as f64, p.z as f64]) as f32
}

trait SurfaceFn {
    fn map(&self, p: Vec3) -> f32;
    fn get_point(&self, p: Vec3) -> f32;
    fn get_normal(&self, p: Vec3) -> Vec3;
}


struct SurfaceGen {
    fbm: Fbm
}

impl SurfaceGen {
    fn new() -> Self {
        let fbm = { 
            let fbm = Fbm::new();
        use noise::MultiFractal;
        fbm.set_octaves(4)};
        Self {
            fbm: fbm
        }
    }
}

impl SurfaceFn for SurfaceGen {
    fn get_point(&self, p: Vec3) -> f32 {
        self.map(p)
    }

    fn get_normal(&self, p: Vec3) -> Vec3 {
        let eps = 0.000001f32; // or some other value
            //let v0 = Vec3::new( -1.0,-1.0,-1.0);
            let v1 = Vec3::new( 1.0,-1.0,-1.0);
            let v2 = Vec3::new(-1.0,-1.0, 1.0);
            let v3 = Vec3::new(-1.0, 1.0,-1.0);
            let v4 = Vec3::new( 1.0, 1.0, 1.0);

            //let s0 = p + v0*eps;
            let s1 = p + v1*eps;
            let s2 = p + v2*eps;
            let s3 = p + v3*eps;
            let s4 = p + v4*eps;

            //let h0 = self.map(s0);
            let h1 = self.map(s1);
            let h2 = self.map(s2);
            let h3 = self.map(s3);
            let h4 = self.map(s4);
          
            ( v1 * h1 +
            v2 * h2 +
            v3 * h3 +
            v4 * h4 ).normalize()
    }

    fn map(&self, p: Vec3) -> f32 {
        //let p = p * 2f32;
        let a = self.fbm.get([p.x as f64, p.y  as f64,  p.z as f64]) as f32;
        //if a < 0.05f32 { 0.05f32 } else { a }
        a
    }

}

fn calc_normal(p: Vec3) -> Vec3 // for function f(p)
{
    let eps = 0.0001f32; // or some other value
    let h = Vec2::new(eps, 0f32);
    return Vec3::new(
        f(p + Vec3::new(h.x, h.y, h.y)) - f(p - Vec3::new(h.x, h.y, h.y)),
        f(p + Vec3::new(h.y, h.x, h.y)) - f(p - Vec3::new(h.y, h.x, h.y)),
        f(p + Vec3::new(h.y, h.y, h.x)) - f(p - Vec3::new(h.y, h.y, h.x)),
    )
    .normalize();
}
/// A square on the XZ plane.
#[derive(Debug, Clone)]
pub struct Quad {
    /// The total side length of the square.
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub d: Vec3,

    pub depth: i32,
}

impl Quad {
    fn new(size: f32, transform: bevy::math::Mat4) -> Self {
        let extent = size / 2.0;
        let a = Vec3::new(extent, 0f32, -extent);
        let b = Vec3::new(extent, 0f32, extent);
        let c = Vec3::new(-extent, 0f32, extent);
        let d = Vec3::new(-extent, 0f32, -extent);
        Quad {
            a,
            b,
            c,
            d,
            depth: 0,
        }
    }

    fn normal(&self) -> Vec3 {
        let p = self.points();
        return (p[0] - p[1]).cross(p[1] - p[2]);
    }
    fn normal2(&self, t: Mat4) -> Vec3 {
        t.transform_vector3(self.normal()).normalize()
    }

    fn center(&self) -> Vec3 {
        (self.a + self.b + self.c + self.d) / 4.0
    }

    fn points(&self) -> [Vec3; 4] {
        let vertices = [self.a, self.b, self.c, self.d];
        vertices
    }

    fn area(&self) -> f32 {
        let p = self.points();
        let segments = [(p[0], p[1]), (p[1], p[2]), (p[2], p[3])];
        let sides = segments.map(|(x, y)| (x - y).length());
        let s = segments
            .iter()
            .map(|(x, y)| (*x - *y).length())
            .sum::<f32>();
        (s * (s - (sides[0]) * sides[1] * sides[2])).sqrt()
    }
}

impl Node {
    fn to_mesh(&self, transform: Mat4, surface: &dyn SurfaceFn) -> Mesh {
        let node = self;
        let plane = &node.quad;

        //println!("{:?}", plane.borders);
        use noise::MultiFractal;
        let gsize = 64u32;//64u32; //128u32;
        let grid_height = gsize;
        let grid_width = gsize;

       
        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let vertex_number = ((grid_height + 1) * (grid_width + 1)) as usize;

        vertices.resize(vertex_number, [0.0f32, 0.0f32, 0.0f32]);
        normals.resize(vertex_number, [0.0f32, 1.0f32, 0.0f32]);
        let uvs = vec![[0.0, 0.0]; vertices.len()];

        let mut vertex_index = 0;
        for mut cy2 in 0..(grid_height as i32 + 1) {
            for mut cx2 in 0..(grid_width as i32 + 1) {
                let mut height_overide = 1f32;

                let mut cx = cx2 as f32 / grid_width as f32;
                let mut cy = cy2 as f32 / grid_height as f32;
                cx = (cx+0.5) * 1.005f32 - 0.5;
                cy = (cy+0.5) * 1.005f32 - 0.5;


                let p = Vec3::new(
                    plane.d.x * (1f32 - cx) + plane.b.x * cx,
                    0.0,
                    plane.a.z * (1f32 - cy) + plane.c.z * cy as f32,
                );
                //let height = f(p);

                let p2 = p * 0.01f32;

                let vp = Vec3::new(
                    plane.d.x * (1f32 - cx) + plane.b.x * cx,
                    0f32,
                    plane.a.z * (1f32 - cy) + plane.c.z * cy,
                );

                let vp = transform.transform_point3(vp);
                let vp_normal = vp.normalize();
                let vpb = vp_normal * 10f32;
                let height = 18f32 * surface.get_point(vpb);
                vertices[vertex_index] = (vp_normal*(500f32+height)).into();

                // deduce terrain normal
                normals[vertex_index] = (vp_normal - surface.get_normal(vpb)).normalize().into();//.lerp(vp_normal, 0.25).into();

                vertex_index += 1;
            }
        }

        let grid_height = grid_height + 1;
        let grid_width = grid_width + 1;
        for cy in 0..(grid_height - 1) {
            for cx in 0..(grid_width - 1) {
                indices.extend(
                    [
                        cy * grid_width + cx,
                        (cy + 1) * grid_width + cx + 1,
                        cy * grid_width + cx + 1,
                    ]
                    .iter(),
                );
                indices.extend(
                    [
                        cy * grid_width + cx,
                        (cy + 1) * grid_width + cx,
                        (cy + 1) * grid_width + cx + 1,
                    ]
                    .iter(),
                );
            }
        }
        //assert!(indices.len() as u32 /  3 == 2  * grid_height * (grid_width) );
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_indices(Some(Indices::U32(indices)));
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh
    }
}

#[derive(Debug)]
struct Node {
    quad: Quad,
    children: Option<[Quad; 4]>,
    address: String,
    parent: String, //leafs: Option<Vec<Box<Self>>>
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            quad: self.quad.clone(),
            address: self.address.clone(),
            parent: self.parent.clone(),
            children: self.children.clone(),
        }
    }
}

impl Node {
    fn subdivide(&self) -> [Node; 4] {
        let mid = self.quad.center();

        let mut nw = Quad {
            a: Vec3::new(mid.x, 0f32, mid.z),
            b: Vec3::new(mid.x, 0f32, self.quad.b.z),
            c: Vec3::new(self.quad.c.x, 0f32, self.quad.c.z),
            d: Vec3::new(self.quad.d.x, 0f32, mid.z),
            depth: self.quad.depth + 1,
        };
        let mut ne = Quad {
            a: Vec3::new(self.quad.a.x, 0f32, mid.z),
            b: Vec3::new(self.quad.b.x, 0f32, self.quad.b.z),
            c: Vec3::new(mid.x, 0f32, self.quad.c.z),
            d: Vec3::new(mid.x, 0f32, mid.z),
            depth: self.quad.depth + 1,
        };
        let mut se = Quad {
            a: Vec3::new(self.quad.a.x, 0f32, self.quad.a.z),
            b: Vec3::new(self.quad.b.x, 0f32, mid.z),
            c: Vec3::new(mid.x, 0f32, mid.z),
            d: Vec3::new(mid.x, 0f32, self.quad.d.z),

            depth: self.quad.depth + 1,
        };
        let mut sw = Quad {
            a: Vec3::new(mid.x, 0f32, self.quad.a.z),
            b: Vec3::new(mid.x, 0f32, mid.z),
            c: Vec3::new(self.quad.c.x, 0f32, mid.z),
            d: Vec3::new(self.quad.d.x, 0f32, self.quad.d.z),
            depth: self.quad.depth + 1,
        };

        let mid = (self.address.clone(), self.quad.center());

        [
            Node {
                children: None,
                quad: nw,
                address: self.address.clone() + "0",
                parent: self.address.clone(),
            },
            Node {
                children: None,
                quad: ne,
                address: self.address.clone() + "1",
                parent: self.address.clone(),
            },
            Node {
                children: None,
                quad: se,
                address: self.address.clone() + "2",
                parent: self.address.clone(),
            },
            Node {
                children: None,
                quad: sw,
                address: self.address.clone() + "3",
                parent: self.address.clone(),
            },
        ]
    }
}

#[derive(Component)]
struct TileTag {
    address: String,
}

enum TileOp {
    Remove(String),
    Add((String, Mesh)),
}

struct NodeTree {
    transform: bevy::math::Mat4,
    nodes: std::collections::HashMap<String, Node>,
    remove_list: Vec<String>,
    insert_list: Vec<(String, Node)>,
}

struct Planet {
    trees: [NodeTree; 6],
    meshes: HashMap<String, Handle<Mesh>>
}

impl Planet {
    fn new() -> Self {
        let pi = 3.14159f32;
        let pih = pi / 2f32;
        let tx_offset = bevy::math::Mat4::from_translation(Vec3::new(0f32, 500f32, 0f32));
        let tx_top = bevy::math::Mat4::from_axis_angle(Vec3::X, 0f32) * tx_offset;
        let tx_right = bevy::math::Mat4::from_axis_angle(Vec3::X, pih) * tx_offset;
        let tx_left = bevy::math::Mat4::from_axis_angle(Vec3::X, -pih) * tx_offset;
        let tx_down = bevy::math::Mat4::from_axis_angle(Vec3::X, pih) * tx_offset;
        let tx_front = bevy::math::Mat4::from_axis_angle(Vec3::Z, pih) * tx_offset;
        let tx_back = bevy::math::Mat4::from_axis_angle(Vec3::Z, -pih) * tx_offset;


        Self {
            trees: [
                NodeTree::new(tx_top),
                NodeTree::new(tx_right),
                NodeTree::new(tx_left),
                NodeTree::new(tx_down),
                NodeTree::new(tx_front),
                NodeTree::new(tx_back),
            ],
            meshes: HashMap::new()
        }
    }

    fn spawn_tasks(
        &mut self,
        commands: &mut Commands,
        campos: Vec3,
        camdir: Vec3,
        thread_pool: Res<AsyncComputeTaskPool>,
        existing_entities: &Vec<String>,
        transform_tasks: Query<(Entity, &TileTag)>,
    ) {
        for nodetree in &mut self.trees {
            nodetree.nodes.clear();
        }
        let pi = 3.14159f32;
        let pih = pi / 2f32;
        let tx_offset = bevy::math::Mat4::from_translation(Vec3::new(0f32, 500f32, 0f32));
        let tx_top = bevy::math::Mat4::from_axis_angle(Vec3::X, 0f32) * tx_offset;
        let tx_right = bevy::math::Mat4::from_axis_angle(Vec3::X, pih) * tx_offset;
        let tx_left = bevy::math::Mat4::from_axis_angle(Vec3::X, -pih) * tx_offset;
        let tx_down = bevy::math::Mat4::from_axis_angle(Vec3::X, pih) * tx_offset;
        let tx_front = bevy::math::Mat4::from_axis_angle(Vec3::Z, pih) * tx_offset;
        let tx_back = bevy::math::Mat4::from_axis_angle(Vec3::Z, -pih) * tx_offset;
        /*
           let tx_top = bevy::math::Mat4::from_axis_angle(Vec3::X, 0f32) * tx_offset;
           let tx_right = bevy::math::Mat4::from_axis_angle(Vec3::X, 90f32) * tx_offset;
           let tx_left = bevy::math::Mat4::from_axis_angle(Vec3::X, -90f32) * tx_offset;
           let tx_down = bevy::math::Mat4::from_axis_angle(Vec3::X, 180f32) * tx_offset;
           let tx_front = bevy::math::Mat4::from_axis_angle(Vec3::Z, 90f32) * tx_offset;
           let tx_back = bevy::math::Mat4::from_axis_angle(Vec3::Z, -90f32) * tx_offset;
        */

        let node = Node {
            quad: Quad::new(1000.0, tx_top),
            address: "A".to_string(),
            parent: "?".to_string(),
            children: None,
        };
        self.trees[0].nodes.insert(node.address.clone(), node);
        let node = Node {
            quad: Quad::new(1000.0, tx_right),
            address: "B".to_string(),
            parent: "?".to_string(),
            children: None,
        };
        self.trees[1].nodes.insert(node.address.clone(), node);
        let node = Node {
            quad: Quad::new(1000.0, tx_left),
            address: "C".to_string(),
            parent: "?".to_string(),
            children: None,
        };
        self.trees[2].nodes.insert(node.address.clone(), node);
        let node = Node {
            quad: Quad::new(1000.0, tx_down),
            address: "D".to_string(),
            parent: "?".to_string(),
            children: None,
        };
        self.trees[3].nodes.insert(node.address.clone(), node);
        let node = Node {
            quad: Quad::new(1000.0, tx_front),
            address: "E".to_string(),
            parent: "?".to_string(),
            children: None,
        };
        self.trees[4].nodes.insert(node.address.clone(), node);
        let node = Node {
            quad: Quad::new(1000.0, tx_back),
            address: "F".to_string(),
            parent: "?".to_string(),
            children: None,
        };
        self.trees[5].nodes.insert(node.address.clone(), node);

        println!("steping...");
            for nodetree in &mut self.trees {
                for _ in 0..10 {
                    nodetree.step(&|world_position| {
                    let dist = (campos - world_position).length();
                    dist
                }, &|cn| { cn.dot(camdir) > 0.0 });
                nodetree.cleanup();
            }
        }


        for nodetree in &mut self.trees {
                let tx = nodetree.transform;
                for (k, n) in &nodetree.nodes {


                    if n.quad.normal2(tx).dot(camdir) < -0.5 { continue  }


                    let surface = SurfaceGen::new();
                    if !existing_entities.contains(&k) {
                        let node = n.clone();
                        let task = thread_pool.spawn(async move {
                            TileOp::Add((node.address.clone(), node.to_mesh(tx, &surface)))
                        });
                        commands.spawn().insert(task);
                    }
                }
            }
            
                for (entity, tiletag) in transform_tasks.iter() {
                        if !self.trees.iter().map(|x|x.nodes.contains_key(&tiletag.address)).fold(false, |a,x| a ||x) {
                        let a = tiletag.address.clone();
                        let task = thread_pool.spawn(async move {
                            TileOp::Remove(a)
                        });
                        commands.spawn().insert(task);
                    }
                }

                        
                for nodetree in &mut self.trees {

            nodetree.remove_list.clear();
        }
    }
}

impl NodeTree {
    fn new(transform: bevy::math::Mat4) -> Self {
        Self {
            transform,
            nodes: std::collections::HashMap::new(),
            remove_list: vec![],
            insert_list: vec![],
        }
    }
    fn step<F, Fnorm>(&mut self, func: &F, func_norm: &Fnorm)
    //where F: Fn(Vec3) -> Option<Vec2>
    where
        F: Fn(Vec3) -> f32,
        Fnorm: Fn(Vec3) -> bool,
    {
        for (k, n) in self.nodes.iter() {
            //let draw = func_norm(n.quad.normal());

            let tpoints = n.quad.points().to_vec().iter().map(|p| func(self.transform.transform_point3(*p/2f32)) ).reduce(f32::min).unwrap();
            let cntr = (self.transform.transform_point3(n.quad.a) 
                        + self.transform.transform_point3(n.quad.b) + self.transform.transform_point3(n.quad.c) 
                        + self.transform.transform_point3(n.quad.d)) / 4.0;
            let cdist = tpoints;//func(cntr);
            let lod = -(cdist / 2500.0).log2();
            let depth = n.address.len() as f32;
            //println!("ID: {}, depth: {}, cdist: {}", n.address, (depth), lod);
            //if depth < (10000f32/(cdist)) {
            if (depth) < lod {
                let quads = n.subdivide();
                for node in quads {
                    let ad = node.address.clone();
                    //println!("ID: {}, dist: {}", node.address, func(node.quad.mid())  );
                    self.insert_list.push((ad, node));
                }
                self.remove_list.push(k.clone());
            }
        }
    }

    fn cleanup(&mut self) {
        for k in self.remove_list.iter() {
            self.nodes.remove(&*k);
        }
        self.remove_list.clear();

        loop {
            match self.insert_list.pop() {
                Some((k, v)) => {
                    if !self.nodes.contains_key(&k) {
                        self.nodes.insert(k, v);
                    }
                }
                None => break,
            }
        }
    }
}
const LABEL: &str = "my_fixed_timestep";

#[derive(Debug, Hash, PartialEq, Eq, Clone, StageLabel)]
struct FixedUpdateStage;
fn main() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        //.insert_resource(AtmosphereMat::default())// Default AtmosphereMat, we can edit it to simulate another planet
        .add_plugins(DefaultPlugins)
        //.add_plugin(AtmospherePlugin { dynamic: true })
        .add_startup_system(setup)
        //.add_startup_system(setup_environment.system())
        //.add_system(daylight_cycle.system())
        .insert_resource(Planet::new())
        //.add_startup_system(spawn_tasks)
        //.add_system_set(
        //    SystemSet::new()
        //        .with_run_criteria(bevy::core::FixedTimestep::step(4.0))
        //        .with_system(spawn_tasks.system()),
        //)
        
        .add_stage_after(
            CoreStage::Update,
            FixedUpdateStage,
            SystemStage::parallel()
                .with_run_criteria(
                    bevy::core::FixedTimestep::step(2.5)
                        // labels are optional. they provide a way to access the current
                        // FixedTimestep state from within a system
                        .with_label(LABEL),
                )
                .with_system(spawn_tasks),
        )
        .add_system(handle_tasks)
        .add_system(animate_cam)
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    windows: Res<Windows>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(AmbientLight {
        color: Color::rgb(0.9, 0.9, 1.0),
        brightness: 0.15,
    });

    let camerab = PerspectiveCameraBundle {
        transform: Transform::from_xyz(560.0 , 560.0, 560.0)
            .looking_at(Vec3::new(540f32, 540f32, 0f32), Vec3::Y),
        ..Default::default()
    };

    let wind = windows.get_primary().unwrap();
    let warea = wind.width() * wind.height();

    commands.spawn_bundle(camerab);

    // light
    const HALF_SIZE: f32 = 1000.0;
    commands.spawn_bundle(DirectionalLightBundle {
        transform: Transform::from_xyz(2.0, 4.0, -4.0).looking_at(Vec3::new(0.0,0.0,0.0), Vec3::new(0.0,1.0,0.0)),
        directional_light: DirectionalLight {
            illuminance: 40_000.0,
            shadow_projection: OrthographicProjection {
                left: -HALF_SIZE,
                right: HALF_SIZE,
                bottom: -HALF_SIZE,
                top: HALF_SIZE,
                near: -10.0 * HALF_SIZE,
                far: 10.0 * HALF_SIZE,
                ..Default::default()
            },
            shadows_enabled: true,
            ..Default::default()
        },
        ..Default::default()
    });
    // camera
}

fn animate_cam(time: Res<Time>, keyboard_input: Res<Input<KeyCode>>,  mut cam: Query<(&mut Transform, With<Camera>)>) {
    //for (mut t, _) in cam.iter_mut() {
    //    t.translation.y =
    //        1f32 + ((1f64 - (time.seconds_since_startup() / 10.0).cos().abs()) * 700.0) as f32; //0.995;25.0;//
    //    t.translation.x =
    //        1f32 + ((1f64 - (time.seconds_since_startup() / 10.0).cos()) * 700.0) as f32;
    //}
        /*
    for (mut t, _) in cam.iter_mut() {
        let z = 240f32 + (1f32-(time.seconds_since_startup() as f32 / 30.0f32).cos().abs()) * 200f32;
        t.translation.y = z; //0.995;25.0;//
        t.translation.x = z;
        t.translation.z = z;
      //      1f32 + ((1f64 - (time.seconds_since_startup() / 10.0).cos()) * 700.0) as f32;
    } */

    let hspeed = if keyboard_input.pressed(KeyCode::LShift) { 50.0f32 } else { 1.0f32} ;

    for (mut t, _) in cam.iter_mut() {
        if keyboard_input.pressed(KeyCode::Space) {
            let tx = t.clone();
            t.translation += (tx.rotation * Vec3::new(0.0,0.0,-2.5 * hspeed * time.delta_seconds()));
        } else if keyboard_input.pressed(KeyCode::Z) {
            let tx = t.clone();
            t.translation += (tx.rotation * Vec3::new(0.0,0.0, 2.5 * hspeed * time.delta_seconds()));
        }

        if keyboard_input.pressed(KeyCode::S) {
            *t = t.mul_transform(Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.01, 0.0, 0.0)));
        } else if keyboard_input.pressed(KeyCode::W) {
            *t = t.mul_transform(Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.01, 0.0, 0.0)));
        }

        if keyboard_input.pressed(KeyCode::A) {
            *t= t.mul_transform(Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, 0.01)));
        } else if keyboard_input.pressed(KeyCode::D) {
            *t = t.mul_transform(Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, -0.01)));
        }
    }
}

fn spawn_tasks(
    mut commands: Commands,
    mut planet: ResMut<Planet>,
    thread_pool: Res<AsyncComputeTaskPool>,
    cam: Query<(&Transform, &Camera)>,
    transform_tasks: Query<(Entity, &TileTag)>,
) {
    let campos = cam.single().0.translation;
    let camdir = cam.single().0.forward();

    let existing_entities = transform_tasks
        .iter()
        .map(|(e, tt)| tt.address.clone())
        .collect::<Vec<_>>();

    planet.spawn_tasks(
        &mut commands,
        campos, camdir,
        thread_pool,
        &existing_entities,
        transform_tasks,
    );
}
use rand::Rng;
fn handle_tasks(
    mut commands: Commands,
    mut planet: ResMut<Planet>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut transform_tasks: Query<(Entity, &mut Task<TileOp>)>,
    cleanups: Query<(Entity, &TileTag)>,
) {
    let mut rng = rand::thread_rng();
    let mut rl = vec![];
    for (entity, mut task) in transform_tasks.iter_mut() {
        if let Some(top) = future::block_on(future::poll_once(&mut *task)) {
            match top {
                TileOp::Remove(id) => {
                    if !planet.trees.iter().map(|n| n.nodes.contains_key(&id)).fold(false, |a,x| a || x) {
                        rl.push(id.clone());
                    }
                }
                TileOp::Add((address, mesh)) => {
                    let mhandle = meshes.add(mesh);
                    planet.meshes.insert(address.clone(), mhandle.clone());

                    commands
                        .spawn_bundle(PbrBundle {
                            mesh: mhandle,
                            //material: materials.add(Color::rgb(rng.gen(), rng.gen(), rng.gen()).into()),
                            material: materials.add( StandardMaterial {
                                base_color: Color::rgb(0.9, 0.9, 0.9).into(),
                                perceptual_roughness: 0.99,
                                metallic: 0.01,
                                reflectance: 0.01,
                                ..Default::default()
                            }),
                            ..Default::default()
                        })
                        //.insert(Wireframe)
                        .insert(TileTag { address });
                }
            }

            // Task is complete, so remove task component from entity
            commands.entity(entity).remove::<Task<TileOp>>();
        }
    }

    for (entity, tiletag) in cleanups.iter() {
        //println!("cleaning up1 {} {:?}", tiletag.address, nodetree.remove_list);
        if rl.contains(&tiletag.address) {
            //println!("cleaning up {}", tiletag.address);
            commands.entity(entity).despawn();
            let mesh = planet.meshes.get(&tiletag.address).unwrap();
            meshes.remove(mesh);
        }
    }
}