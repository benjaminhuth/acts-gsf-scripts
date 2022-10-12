import pyg4ometry

def make_gdml(filename, n_surfaces, surface_width, surface_thickness, surface_distance, visualize=False):
    # registry to store gdml data
    reg = pyg4ometry.geant4.Registry()

    surface_width = pyg4ometry.gdml.Constant("w", surface_width, reg)
    surface_thickness = pyg4ometry.gdml.Constant("t", surface_thickness, reg)
    # make the world twice as long to be able to move the volumes around
    detector_length = pyg4ometry.gdml.Constant("l", n_surfaces * surface_distance * 2, reg)

    world_material = pyg4ometry.geant4.MaterialPredefined("G4_Galactic", reg)
    surface_material = pyg4ometry.geant4.MaterialPredefined("G4_Si", reg)

    # world solid and logical
    world = pyg4ometry.geant4.solid.Box("world",
                                    detector_length,
                                    surface_width,
                                    surface_width,
                                    reg)

    world_logical = pyg4ometry.geant4.LogicalVolume(world,
                                                    world_material,
                                                    "world_logical",
                                                    reg)

    reg.setWorld(world_logical)

    # Surfaces
    for i in range(n_surfaces):
        s = pyg4ometry.geant4.solid.Box("s{}".format(i),
                                        surface_thickness,
                                        surface_width,
                                        surface_width,
                                        reg)

        s_logical = pyg4ometry.geant4.LogicalVolume(s,
                                                    surface_material,
                                                    "s{}_l".format(i),
                                                    reg)

        x_pos = i * surface_distance
        s_physical = pyg4ometry.geant4.PhysicalVolume([0,0,0],
                                                    [x_pos,0,0],
                                                    s_logical,
                                                    "s{}_p".format(i),
                                                    world_logical,
                                                    reg)

    # export
    w = pyg4ometry.gdml.Writer()
    w.addDetector(reg)
    w.write(filename)

    # visualise geometry
    if visualize:
        v = pyg4ometry.visualisation.VtkViewer()
        v.addLogicalVolume(world_logical)
        v.addAxes(20)
        v.view()

if __name__ == "__main__":
    make_gdml("/dev/null", n_surfaces=10, surface_width=10, surface_thickness=1, surface_distance=5, visualize=True)
