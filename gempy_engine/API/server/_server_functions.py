import sys
import logging

from gempy_engine.core.data.engine_grid import EngineGrid
from core.data.regular_grid import RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.server.input_parser import GemPyInput


def setup_logger():
    logger = logging.getLogger("my-fastapi-app")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Set up console handler for logging
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # Set up file handler for logging
    fh = logging.FileHandler("app.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    # Add the console handler to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def process_input(gempy_input: GemPyInput, logger: logging.Logger) -> (InputDataDescriptor, InterpolationInput, int):
    
    logger.debug("Input grid:", gempy_input.interpolation_input.grid)
    
    gempy_input.interpolation_input.grid = EngineGrid.from_regular_grid(
        regular_grid=RegularGrid.from_schema(gempy_input.interpolation_input.grid)
    )
    interpolation_input: InterpolationInput = InterpolationInput.from_schema(gempy_input.interpolation_input)
    input_data_descriptor: InputDataDescriptor = InputDataDescriptor.from_schema(gempy_input.input_data_descriptor)
    n_stack = len(input_data_descriptor.stack_structure.masking_descriptor)
    logger.debug("masking descriptor: ", input_data_descriptor.stack_structure.masking_descriptor)
    logger.debug("stack structure: ", input_data_descriptor.stack_structure)
    return input_data_descriptor, interpolation_input, n_stack

