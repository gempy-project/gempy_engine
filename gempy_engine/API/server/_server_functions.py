import logging
import sys
from typing import Tuple

from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.server.input_parser import GemPyInput


def setup_logger() -> logging.Logger:
    """
    Configure and set up a logger for the application.
    
    Returns:
        logging.Logger: Configured logger instance with console and file handlers
    """
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

    # Add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def process_input(
    gempy_input: GemPyInput, 
    logger: logging.Logger
) -> Tuple[InputDataDescriptor, InterpolationInput, int]:
    """
    Process the GemPy input data to prepare it for model computation.
    
    Args:
        gempy_input: Input data for the GemPy model
        logger: Logger instance for recording processing information
        
    Returns:
        Tuple containing:
            - InputDataDescriptor: Structure descriptor for the model
            - InterpolationInput: Prepared interpolation input data
            - int: Number of stacks in the model
    """
    logger.debug(f"Input grid: {gempy_input.interpolation_input.grid}")

    interpolation_input: InterpolationInput = InterpolationInput.from_schema(gempy_input.interpolation_input)
    input_data_descriptor: InputDataDescriptor = InputDataDescriptor.from_schema(gempy_input.input_data_descriptor)
    n_stack = len(input_data_descriptor.stack_structure.masking_descriptor)

    logger.debug(f"masking descriptor: {input_data_descriptor.stack_structure.masking_descriptor}")
    logger.debug(f"stack structure: {input_data_descriptor.stack_structure}")
    
    return input_data_descriptor, interpolation_input, n_stack