/// MFEM Example 1
///
/// This example code demonstrates the use of MFEM to define a simple
/// finite element discretization of the Laplace problem -Δu = 1 with
/// homogeneous Dirichlet boundary conditions.  Specifically, we
/// discretize using a FE space of the specified order, or if order
/// < 1 using an isoparametric/isogeometric space (i.e., quadratic for
/// quadratic curvilinear mesh, NURBS for NURBS mesh, etc.)
///
/// The example highlights the use of mesh refinement, finite element
/// grid functions, as well as linear and bilinear forms corresponding
/// to the left-hand side and right-hand side of the discrete linear
/// system. We also cover the explicit elimination of essential
/// boundary conditions, static condensation, and the optional
/// connection to the GLVis tool for visualization.
use clap::Parser;
use mfem::*;

#[derive(Parser)]
#[command(version)]
struct Args {
    /// Mesh file to use.
    #[arg(short, long = "mesh", value_name = "FILE")]
    mesh_file: String,

    /// Finite element order (polynomial degree) or -1 for isoparametric space.
    #[arg(short, long, default_value_t = 1)]
    order: i32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Parse command-line options.
    let args = Args::parse();

    // 2. Enable hardware devices such as GPUs, and programming models
    //    such as CUDA, OCCA, RAJA and OpenMP based on command line options.
    // TODO(mkovaxx)

    // 3. Read the mesh from the given mesh file. We can handle
    //    triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume meshes with the same code.
    let mut mesh = Mesh::from_file(&args.mesh_file)?;
    let dim = mesh.dimension();
    dbg!(dim);
    dbg!(mesh.get_num_elems());

    // 4. Refine the mesh to increase the resolution.  In this example
    //    we do `ref_levels` of uniform refinement.  We choose
    //    `ref_levels` to be the largest number that gives a final
    //    mesh with no more than 50,000 elements.
    let ne = mesh.get_num_elems() as f64;
    let ref_levels = ((50000. / ne).log2() / dim as f64).floor() as u32;
    for _ in 0..ref_levels {
        mesh.uniform_refinement(RefAlgo::A);
    }
    dbg!(mesh.get_num_elems());

    // 5. Define a finite element space on the mesh. Here we use
    //    continuous Lagrange finite elements of the specified order.
    //    If order < 1, we instead use an isoparametric/isogeometric space.
    let nodes = mesh.get_nodes();
    let fec: &FiniteElementCollectionBase = if args.order > 0 {
        &H1_FECollection::new(args.order, dim, BasisType::GaussLobatto)
    } else if nodes.is_some() {
        &nodes.unwrap().fec()
    } else {
        &H1_FECollection::new(1, dim, BasisType::GaussLobatto)
    };
    dbg!(fec.get_name());

    let fespace = FiniteElementSpace::new(&mesh, fec, 1, Ordering::ByNodes);
    println!(
        "Number of finite element unknowns: {}",
        fespace.get_true_vsize(),
    );

    // 6. Determine the list of true (i.e. conforming) essential
    //    boundary dofs.  In this example, the boundary conditions are
    //    defined by marking all the boundary attributes from the mesh
    //    as essential (Dirichlet) and converting them to a list of
    //    true dofs.
    let mut ess_tdof_list = ArrayInt::new();
    if let Some(max_bdr_attr) = mesh.bdr_attributes().iter().max() {
        let mut ess_bdr = ArrayInt::with_len(*max_bdr_attr as usize);
        ess_bdr.fill(1);
        fespace.get_essential_true_dofs(&ess_bdr, &mut ess_tdof_list, None);
    }

    // // 7. Set up the linear form b(.) which corresponds to the
    // //    right-hand side of the FEM linear system, which in this case
    // //    is (1,phi_i) where phi_i are the basis functions in the
    // //    finite element fespace.
    let mut b = LinearForm::new(&fespace);
    let mut one = ConstantCoefficient::new(1.0);
    let integrator = DomainLFIntegrator::new(&mut one, 2, 0);
    b.add_domain_integrator(integrator);
    // drop(one); // With this, it must fail to compile.
    b.assemble();

    // 8. Define the solution vector `x` as a finite element grid
    //    function corresponding to fespace.  Initialize x with initial
    //    guess of zero, which satisfies the boundary conditions.
    let mut x = GridFunction::with_fes(&fespace);
    x.fill(0.0);

    // 9. Set up the bilinear form a(.,.) on the finite element space
    //    corresponding to the Laplacian operator -Delta, by adding
    //    the Diffusion domain integrator.
    let mut a = BilinearForm::new(&fespace);
    // Remark: the original code reuses the `one` above but, since it
    // is mutably borrowed and still needed for `b`, a new coefficient
    // is better.
    let mut one = ConstantCoefficient::new(1.0);
    let bf_integrator = DiffusionIntegrator::new(&mut one);
    a.add_domain_integrator(bf_integrator);

    // 10. Assemble the bilinear form and the corresponding linear
    //     system, applying any necessary transformations such as:
    //     eliminating boundary conditions, applying conforming
    //     constraints for non-conforming AMR, static condensation, etc.
    a.assemble(true);

    // let mut a_mat = OperatorHandle::new();
    // let mut b_vec = Vector::new();
    // let mut x_vec = Vector::new();
    // a.form_linear_system(&ess_tdof_list, &x, &b,
    //     &mut a_mat, &mut x_vec, &mut b_vec);

    // println!("Size of linear system: {}", a_mat.height());
    // dbg!(a_mat.get_type());

    // // 11. Solve the linear system A X = B.
    // // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
    // let a_sparse = SparseMatrixRef::try_from(&a_mat).expect("Operator is a SparseMatrix");
    // let mut m_mat = GsSmoother::new(&a_sparse, 0, 1);
    // solve_with_pcg(&a_mat, &mut m_mat, &b_vec, &mut x_vec, 1, 200, 1e-12, 0.0);

    // 12. Recover the solution as a finite element grid function.
    // a.recover_fem_solution(&x_vec, &b, &mut x);

    // 13. Save the refined mesh and the solution. This output can be
    //     viewed later using GLVis: "glvis -m refined.mesh -g sol.gf".
    mesh.save().to_file("refined.mesh");
    // x.save_to_file("sol.gf", 8);


    Ok(())
}
