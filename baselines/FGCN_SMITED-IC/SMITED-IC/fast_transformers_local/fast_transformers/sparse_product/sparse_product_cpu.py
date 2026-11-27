"""
Stub for sparse_product/sparse_product_cpu compiled extension.
This stub allows imports to succeed without compilation.
The actual functions will raise errors if called.

Note: SMI-TED uses linear attention which doesn't require these compiled extensions.
"""

def __getattr__(name):
    """Lazy stub - any function call will raise an error."""
    def stub_function(*args, **kwargs):
        raise NotImplementedError(
            f"{name} requires compiled C++/CUDA extensions. "
            f"This is a stub to allow imports. "
            f"SMI-TED uses linear attention which doesn't need this function."
        )
    return stub_function
