#!/usr/bin/env python3

import argparse
from pathlib import Path

import gmsh
import numpy as np


def generate_unstructured_quad_cylinder_mesh(
    length=1.0,
    radius=0.5,
    target_size=0.12,
    n_interior_pts=80,
    seed=1,
    crazy=False,
):
    """Generate an unstructured quad-dominant cylindrical surface mesh."""
    if length <= 0.0:
        raise ValueError("length must be positive")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if target_size <= 0.0:
        raise ValueError("target_size must be positive")
    if n_interior_pts < 0:
        raise ValueError("n_interior_pts cannot be negative")

    rng = np.random.default_rng(seed)
    circumference = 2.0 * np.pi * radius

    gmsh.initialize()
    try:
        gmsh.model.add("unstructured_quad_cylinder")

        # Mesh the unwrapped cylinder in (x, s), where s is arc length.
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, target_size)
        p2 = gmsh.model.geo.addPoint(length, 0.0, 0.0, target_size)
        p3 = gmsh.model.geo.addPoint(length, circumference, 0.0, target_size)
        p4 = gmsh.model.geo.addPoint(0.0, circumference, 0.0, target_size)

        bottom = gmsh.model.geo.addLine(p1, p2)
        right = gmsh.model.geo.addLine(p2, p3)
        top = gmsh.model.geo.addLine(p3, p4)
        left = gmsh.model.geo.addLine(p4, p1)

        curve_loop = gmsh.model.geo.addCurveLoop([bottom, right, top, left])
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # Random embedded points make the mesh genuinely unstructured.
        margin_x = 0.05 * length
        margin_s = 0.05 * circumference
        interior_point_tags = []

        if crazy:
            # More points, clustered placement, and strongly varying local sizes.
            effective_points = 4 * n_interior_pts
            nclusters = max(3, int(np.sqrt(max(1, n_interior_pts)) // 2))

            cluster_x = rng.uniform(
                margin_x, length - margin_x, size=nclusters
            )
            cluster_s = rng.uniform(
                margin_s, circumference - margin_s, size=nclusters
            )

            for _ in range(effective_points):
                if rng.random() < 0.75:
                    ic = rng.integers(0, nclusters)
                    x = rng.normal(cluster_x[ic], 0.07 * length)
                    s = rng.normal(cluster_s[ic], 0.07 * circumference)

                    x = np.clip(x, margin_x, length - margin_x)
                    s = np.clip(s, margin_s, circumference - margin_s)
                else:
                    x = rng.uniform(margin_x, length - margin_x)
                    s = rng.uniform(margin_s, circumference - margin_s)

                local_size = target_size * rng.uniform(0.25, 2.5)

                interior_point_tags.append(
                    gmsh.model.geo.addPoint(
                        x, s, 0.0, local_size
                    )
                )

            print(
                f"Crazy mesh mode: {effective_points} embedded points, "
                f"{nclusters} clusters, local size range "
                f"[{0.25 * target_size:.4g}, "
                f"{2.5 * target_size:.4g}]"
            )

        else:
            for _ in range(n_interior_pts):
                x = rng.uniform(margin_x, length - margin_x)
                s = rng.uniform(margin_s, circumference - margin_s)

                interior_point_tags.append(
                    gmsh.model.geo.addPoint(
                        x, s, 0.0, target_size
                    )
                )

        gmsh.model.geo.synchronize()

        if interior_point_tags:
            gmsh.model.mesh.embed(0, interior_point_tags, 2, surface)

        # Make top and bottom periodic in the circumferential direction.
        periodic_transform = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, circumference,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        gmsh.model.mesh.setPeriodic(
            1, [top], [bottom], periodic_transform
        )

        # Delaunay triangles followed by Blossom recombination.
        gmsh.model.mesh.setRecombine(2, surface)
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        planar_xyz = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        tag_to_raw_index = {
            int(tag): i for i, tag in enumerate(node_tags)
        }

        element_types, _, element_node_tags = gmsh.model.mesh.getElements(dim=2)

        raw_quad_connectivity = []
        quad_count = 0
        triangle_count = 0

        for element_type, nodes in zip(element_types, element_node_tags):
            name, _, _, num_nodes, _, _ = (
                gmsh.model.mesh.getElementProperties(element_type)
            )
            nodes = np.asarray(nodes, dtype=np.int64)

            if num_nodes == 4 and "Quadrilateral" in name:
                quads = nodes.reshape(-1, 4)
                raw_quad_connectivity.extend(
                    [
                        [tag_to_raw_index[int(tag)] for tag in quad]
                        for quad in quads
                    ]
                )
                quad_count += len(quads)

            elif num_nodes == 3 and "Triangle" in name:
                triangle_count += len(nodes.reshape(-1, 3))

        if not raw_quad_connectivity:
            raise RuntimeError("Gmsh did not generate any four-node quads.")

        raw_quad_connectivity = np.asarray(
            raw_quad_connectivity, dtype=np.int64
        )

    finally:
        gmsh.finalize()

    # Map the unwrapped mesh onto the cylinder.
    x = planar_xyz[:, 0]
    s = np.mod(planar_xyz[:, 1], circumference)

    seam_tolerance = 1.0e-10 * max(1.0, circumference)
    s[np.abs(s) < seam_tolerance] = 0.0
    s[np.abs(s - circumference) < seam_tolerance] = 0.0

    theta = s / radius
    raw_xyz = np.column_stack(
        (
            x,
            radius * np.sin(theta),
            radius * np.cos(theta),
        )
    )

    xyz, conn = merge_duplicate_nodes(
        raw_xyz, raw_quad_connectivity
    )

    print(
        f"Generated {quad_count} quads and "
        f"{triangle_count} leftover triangles."
    )
    print(
        f"Cylinder mesh: {len(xyz)} nodes, "
        f"{len(conn)} CQUAD4 elements."
    )

    if triangle_count:
        print("Leftover triangles were omitted from the BDF.")

    return xyz, conn


def merge_duplicate_nodes(xyz, conn, tolerance=1.0e-10):
    """Merge duplicate seam nodes and update element connectivity."""
    used_nodes = np.unique(conn.ravel())
    raw_to_merged = np.full(len(xyz), -1, dtype=np.int64)

    merged_xyz = []
    key_to_index = {}
    scale = 1.0 / tolerance

    for raw_index in used_nodes:
        point = xyz[raw_index]
        key = tuple(np.rint(point * scale).astype(np.int64))

        if key not in key_to_index:
            key_to_index[key] = len(merged_xyz)
            merged_xyz.append(point.copy())

        raw_to_merged[raw_index] = key_to_index[key]

    merged_conn = raw_to_merged[conn]
    merged_xyz = np.asarray(merged_xyz, dtype=float)

    valid = np.array(
        [len(set(element.tolist())) == 4 for element in merged_conn],
        dtype=bool,
    )

    if not np.all(valid):
        print(
            f"WARNING: removing {np.count_nonzero(~valid)} "
            "degenerate quads after seam merging."
        )
        merged_conn = merged_conn[valid]

    return merged_xyz, merged_conn


# ---------------------------------------------------------------------
# Nastran large-field formatting, matching the uploaded BDF style.
# ---------------------------------------------------------------------

def field8(value=""):
    """Format an 8-character Nastran field."""
    return f"{str(value):<8}"[:8]


def field16(value=""):
    """Format a 16-character Nastran large field."""
    return f"{str(value):>16}"[:16]


def nastran_float16(value):
    """Compact a floating-point value into a 16-character Nastran field."""
    if value == 0.0:
        return "0."

    # Fixed/general notation is easier to read when it fits.
    text = f"{value:.13g}"
    if len(text) <= 16:
        return text

    # Fall back to scientific notation.
    for precision in range(9, 0, -1):
        text = f"{value:.{precision}e}"
        if len(text) <= 16:
            return text

    raise ValueError(f"Cannot fit value {value} in a 16-character field")


def write_grid_large(bdf, nid, x, y, z):
    """Write a GRID* card using two large-field lines."""
    bdf.write(
        f"{field8('GRID*')}"
        f"{field16(nid)}"
        f"{field16('')}"
        f"{field16(nastran_float16(x))}"
        f"{field16(nastran_float16(y))}\n"
    )
    bdf.write(
        f"{field8('*')}"
        f"{field16(nastran_float16(z))}\n"
    )


def write_cquad4_large(bdf, eid, pid, g1, g2, g3, g4):
    """Write a CQUAD4* card using two large-field lines."""
    bdf.write(
        f"{field8('CQUAD4*')}"
        f"{field16(eid)}"
        f"{field16(pid)}"
        f"{field16(g1)}"
        f"{field16(g2)}\n"
    )
    bdf.write(
        f"{field8('*')}"
        f"{field16(g3)}"
        f"{field16(g4)}\n"
    )


def write_pshell_large(bdf, pid, mid, thickness):
    """Write the required PSHELL fields in large-field form."""
    bdf.write(
        f"{field8('PSHELL*')}"
        f"{field16(pid)}"
        f"{field16(mid)}"
        f"{field16(nastran_float16(thickness))}\n"
    )
    bdf.write(f"{field8('*')}\n")


def write_mat1_large(bdf, mid, youngs_modulus, poisson_ratio, density):
    """Write MAT1 with E, NU, and RHO; G is left blank."""
    bdf.write(
        f"{field8('MAT1*')}"
        f"{field16(mid)}"
        f"{field16(nastran_float16(youngs_modulus))}"
        f"{field16('')}"
        f"{field16(nastran_float16(poisson_ratio))}\n"
    )
    bdf.write(
        f"{field8('*')}"
        f"{field16(nastran_float16(density))}\n"
    )


def write_spc1_cards(bdf, sid, components, node_ids, nodes_per_card=6):
    """Write SPC1 cards, splitting long node lists across cards."""
    for start in range(0, len(node_ids), nodes_per_card):
        chunk = node_ids[start : start + nodes_per_card]
        line = (
            f"{field8('SPC1')}"
            f"{field8(sid)}"
            f"{field8(components)}"
            + "".join(field8(nid) for nid in chunk)
        )
        bdf.write(line.rstrip() + "\n")


def write_nastran_bdf(
    filename,
    xyz,
    conn,
    thickness=1.0e-3,
    youngs_modulus=70.0e9,
    poisson_ratio=0.3,
    density=0.0,
    property_id=1,
    material_id=1,
    spc_id=3,
    clamp_x0=True,
):
    """
    Write a large-field Nastran BDF matching the uploaded file's style.

    The x=0 cylinder edge is clamped with SPC1 components 123456 by default.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (nnodes, 3)")
    if conn.ndim != 2 or conn.shape[1] != 4:
        raise ValueError("conn must have shape (nelems, 4)")
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")

    with filename.open("w", encoding="utf-8") as bdf:
        bdf.write("$pyNastran: version=msc\n")
        bdf.write("$pyNastran: punch=False\n")
        bdf.write("$pyNastran: encoding=utf-8\n")
        bdf.write(f"$pyNastran: nnodes={len(xyz)}\n")
        bdf.write(f"$pyNastran: nelements={len(conn)}\n")
        bdf.write("$EXECUTIVE CONTROL DECK\n")
        bdf.write("ID Unstructured Cylinder FOR TACS\n")
        bdf.write("SOL 1\n")
        bdf.write("CEND\n")
        bdf.write("$CASE CONTROL DECK\n")
        bdf.write("DISPLACEMENT(PRINT,PUNCH) = ALL\n")
        bdf.write("LINE = 10000\n")
        bdf.write("STRAIN(PRINT,PUNCH) = ALL\n")
        bdf.write("STRESS(PRINT,PUNCH) = ALL\n")
        bdf.write("SUBCASE 1\n")
        bdf.write("    ANALYSIS = STATICS\n")
        bdf.write("    LABEL = Cylinder\n")
        if clamp_x0:
            bdf.write(f"    SPC = {spc_id}\n")
        bdf.write("BEGIN BULK\n")

        bdf.write("$PARAMS\n")
        bdf.write(
            "$---1---|---2---|---3---|---4---|---5---|"
            "---6---|---7---|---8---|---9---|---10--|\n"
        )
        bdf.write("PARAM*              POST              -1\n")
        bdf.write("*\n")

        bdf.write("$NODES\n")
        bdf.write(
            "$---1A--|-------2-------|-------3-------|"
            "-------4-------|-------5-------|-10A--|\n"
        )
        bdf.write(
            "$---1B--|-------6-------|-------7-------|"
            "-------8-------|-------9-------|-10B--|\n"
        )

        for nid, (x, y, z) in enumerate(xyz, start=1):
            write_grid_large(bdf, nid, x, y, z)

        bdf.write("$ELEMENTS\n")
        bdf.write("$ Property : cylindrical shell\n")

        for eid, element in enumerate(conn, start=1):
            g1, g2, g3, g4 = (element + 1).tolist()
            write_cquad4_large(
                bdf, eid, property_id, g1, g2, g3, g4
            )

        bdf.write("$PROPERTIES\n")
        bdf.write("$ Property : isotropic cylindrical shell\n")
        write_pshell_large(
            bdf, property_id, material_id, thickness
        )

        bdf.write("$MATERIALS\n")
        write_mat1_large(
            bdf,
            material_id,
            youngs_modulus,
            poisson_ratio,
            density,
        )

        if clamp_x0:
            x_tolerance = 1.0e-9 * max(
                1.0, np.max(xyz[:, 0]) - np.min(xyz[:, 0])
            )
            clamped_nodes = (
                np.where(np.abs(xyz[:, 0]) <= x_tolerance)[0] + 1
            ).tolist()

            if not clamped_nodes:
                raise RuntimeError(
                    "No x=0 nodes found for the requested clamp."
                )

            bdf.write("$SPCs\n")
            write_spc1_cards(
                bdf,
                sid=spc_id,
                components=123456,
                node_ids=clamped_nodes,
            )

            print(
                f"Clamped {len(clamped_nodes)} nodes on x=0 "
                f"using SPC set {spc_id}."
            )

        bdf.write("ENDDATA\n")

    print(f"Wrote BDF file: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an unstructured quad-dominant cylindrical "
            "shell mesh and write a large-field Nastran BDF."
        )
    )

# to make mesh very irregular, high # embedded points, large mesh size (spacing)
# to make mesh high DOF but a bit more regular, low # embedded points, smaller mesh size (h-size)
# if too irregular, then corner violations will persist..
# really irregular 10k node mesh is points=3000, size=0.12
# medium regularity 10k node mesh is points=300, size=0.04 (BEST one cause corner violations goes to zero.. but somehow still not perfect BDDC thin shell conv, investigate)
# more regular ~10k node mesh is points=80, size=0.02
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--size", type=float, default=0.04)
    parser.add_argument("--points", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--crazy",
        action="store_true",
        help=(
            "Make the mesh much more irregular using four times as many "
            "embedded points, clustered point placement, and randomized "
            "local mesh sizes."
        ),
    )
    parser.add_argument("--thickness", type=float, default=1.0e-3)
    parser.add_argument("--youngs", type=float, default=70.0e9)
    parser.add_argument("--poisson", type=float, default=0.3)
    parser.add_argument("--density", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unstructured_cylinder.bdf"),
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="Do not write an SPC1 clamp on the x=0 cylinder edge.",
    )

    args = parser.parse_args()

    xyz, conn = generate_unstructured_quad_cylinder_mesh(
        length=args.length,
        radius=args.radius,
        target_size=args.size,
        n_interior_pts=args.points,
        seed=args.seed,
        crazy=args.crazy,
    )

    write_nastran_bdf(
        filename=args.output,
        xyz=xyz,
        conn=conn,
        thickness=args.thickness,
        youngs_modulus=args.youngs,
        poisson_ratio=args.poisson,
        density=args.density,
        clamp_x0=not args.no_clamp,
    )


# to make mesh very irregular, high # embedded points, large mesh size (spacing)
# to make mesh high DOF but a bit more regular, low # embedded points, smaller mesh size (h-size)
# if too irregular, then corner violations will persist..

if __name__ == "__main__":
    main()