from gempy_engine.core.data.finite_fault import TaperType
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.kernel_classes.server.input_parser import InputDataDescriptorSchema


def test_finite_fault_server_schema_converts_to_runtime_data():
    schema = InputDataDescriptorSchema.model_validate({
            "number_of_points_per_surface"      : [3, 3],
            "number_of_points_per_stack"        : [3, 3],
            "number_of_orientations_per_stack"  : [1, 1],
            "number_of_surfaces_per_stack"      : [1, 1],
            "masking_descriptor"                : [3, 1],
            "faults_relations"                  : [[False, True], [False, False]],
            "faults_input_data"                 : [
                    {
                        "finite_fault": {
                            "center"       : [0.0, 0.0, 0.0],
                            "strike_radius": [2.0, 1.0],
                            "dip_radius"   : 0.75,
                            "taper"        : "quadratic",
                            "rotation_deg" : 15.0,
                        }
                    },
                    None,
            ],
    })

    descriptor = InputDataDescriptor.from_schema(schema)
    fault_data = descriptor.stack_structure.faults_input_data[0]

    assert descriptor.stack_structure.faults_relations.tolist() == [[False, True], [False, False]]
    assert fault_data.finite_fault.center == (0.0, 0.0, 0.0)
    assert fault_data.finite_fault.strike_radius == (2.0, 1.0)
    assert fault_data.finite_fault.taper is TaperType.QUADRATIC


def test_finite_fault_server_schema_serializes_as_json_values():
    schema = InputDataDescriptorSchema.model_validate({
            "number_of_points_per_surface"      : [3],
            "number_of_points_per_stack"        : [3],
            "number_of_orientations_per_stack"  : [1],
            "number_of_surfaces_per_stack"      : [1],
            "masking_descriptor"                : [3],
            "faults_input_data"                 : [{"finite_fault": {"center": [1.0, 2.0, 3.0]}}],
    })

    payload = schema.model_dump(mode="json")

    assert payload["faults_input_data"][0]["finite_fault"]["center"] == [1.0, 2.0, 3.0]
    assert payload["faults_input_data"][0]["finite_fault"]["taper"] == "cubic"
