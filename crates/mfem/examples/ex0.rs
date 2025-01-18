//! MFEM Example 0
//!
//! Description: This example code demonstrates the most basic usage of MFEM to
//!              define a simple finite element discretization of the Laplace
//!              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//!              General 2D/3D mesh files and finite element polynomial degrees
//!              can be specified by command line options.
use clap::Parser;
use mfem::*;

#[derive(Parser)]
#[command(version)]
struct Args {
    /// Mesh file to use.
    #[arg(short, long = "mesh", value_name = "FILE", default_value = "data/square.mesh")]
    mesh_file: String,

    /// Finite element order (polynomial degree) or -1 for isoparametric space.
    #[arg(short, long, default_value_t = 1)]
    order: i32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 2. Read the mesh from the given mesh file, and refine once uniformly.
    let mut mesh = Mesh::from_file(&args.mesh_file)?;
    mesh.uniform_refinement(RefAlgo::A);

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
    let fec = H1_FECollection::new(args.order, mesh.dimension());
    let fespace = FiniteElementSpace::new(&mesh, &fec);
    println!("Number of unknowns: {}", fespace.get_true_vsize());

    // 4. Extract the list of all the boundary DOFs. These will be
    //    marked as Dirichlet in order to enforce zero boundary conditions.
    let mut boundary_dofs = ArrayInt::new();
    fespace.get_boundary_true_dofs(&mut boundary_dofs, None);

    // 5. Define the solution x as a finite element grid function in
    //    `fespace`.  Set the initial guess to zero, which also sets the
    //    boundary conditions.
    let mut x = GridFunction::new(&fespace);
    x.fill(0.0);

    // 6. Set up the linear form b(.) corresponding to the right-hand side.
    let mut one = ConstantCoefficient::new(1.0);
    let mut b = LinearForm::new(&fespace);
    b.add_domain_integrator(DomainLFIntegrator::new(&mut one));
    b.assemble();

    // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
    let mut a = BilinearForm::new(&fespace);
    a.add_domain_integrator(DiffusionIntegrator::new());
    a.assemble(true);

    // 8. Form the linear system A X = B. This includes eliminating
    //    boundary conditions, applying AMR constraints, and other
    //    transformations.
    let mut ls = a.form_linear_system(&boundary_dofs, &mut x, &mut b);
    let a_mat: &ASparseMatrix = (&ls.a).try_into()?;

    // 9. Solve the system using PCG with symmetric Gauss-Seidel
    // preconditioner.
    let mut m = GSSmoother::new(&a_mat, 0, 1);
    mfem::pcg(&a_mat, &mut m, &ls.b, &mut ls.x)
        .print_iter(true).solve();

    // 10. Recover the solution x as a grid function and save to
    //     file. The output can be viewed using GLVis as follows:
    //     "glvis -m mesh.mesh -g sol.gf"
    ls.recover_fem_solution();
    x.save().to_file("sol.gf");
    mesh.save().to_file("mesh.mesh");
    Ok(())
}
