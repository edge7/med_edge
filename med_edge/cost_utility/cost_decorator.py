import asyncio
import functools
import os
from typing import Any, Callable

from loguru import logger
from med_edge.cost_utility import COST_INFO, COST_TYPE
from med_edge.cost_utility.cost_writer import push_cost


def log_cost(cost_type: str):
    """
    A decorator that intercepts a dictionary return value to log costs asynchronously.
    This version works for both synchronous and asynchronous functions.

    It expects the decorated function to return a dictionary created by the
    `build_response` helper. It intelligently handles whether the cost value
    is a single item or a list, and submits the cost for logging by calling .delay().
    """


    def decorator(func: Callable[..., Any]):
        # This helper function contains the shared logic for processing the result.
        # It is kept synchronous as it only processes data, not I/O.
        def _process_and_log_cost(result_dict: Any, func_name: str) -> Any:
            # 2. Safely extract data from the result dictionary
            if not isinstance(result_dict, dict) or result_dict is None:
                logger.warning(
                    f"Decorator @log_cost on '{func_name}' did not receive a dictionary. "
                    f"Returning the original result directly. Got: {type(result_dict)}"
                )
                return result_dict

            real_result = result_dict.get("real_result")
            cost_data = result_dict.get("cost")

            # 3. Process the cost data if it exists
            if isinstance(cost_data, dict):
                unique_id = cost_data.get("unique_id")
                cost_value = cost_data.get("value")

                if cost_value is not None and unique_id is not None:
                    # If cost_value is already a list, use it. Otherwise, wrap it.
                    final_cost_list = (
                        cost_value if isinstance(cost_value, list) else [cost_value]
                    )

                    cost_object = {
                        COST_TYPE: cost_type,
                        COST_INFO: final_cost_list,
                        "unique_id": unique_id,
                    }

                    try:
                        # Push cost to the writer queue
                        push_cost(cost_object)
                    except Exception as e:
                        logger.error(
                            f"[{unique_id}] Decorator failed to submit cost to logger: {e}"
                        )
            else:
                logger.error(f"Cost data is not a dict: {cost_data}")

            # 4. Return only the real result to the original caller
            return real_result

        # Check if the decorated function is async
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 1. Execute the actual async function by awaiting it
                result_dict = await func(*args, **kwargs)
                # Process the result and return the real result
                return _process_and_log_cost(result_dict, func.__name__)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # 1. Execute the actual sync function
                result_dict = func(*args, **kwargs)
                # Process the result and return the real result
                return _process_and_log_cost(result_dict, func.__name__)

            return sync_wrapper

    return decorator