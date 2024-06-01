def custom_pykeops_solver(tensor, other, var=None, call=True, **kwargs):
    r"""
    Solves a positive definite linear system of the form ``sum(self) = other`` or ``sum(self*var) = other`` , using a conjugate gradient solver.

    Args:
      self (:class:`LazyTensor`): KeOps variable that encodes a symmetric positive definite matrix / linear operator.
      other (:class:`LazyTensor`): KeOps variable that encodes the second member of the equation.

    Keyword args:
      var (:class:`LazyTensor`):
        If **var** is **None**, **solve** will return the solution
        of the ``self * var = other`` equation.
        Otherwise, if **var** is a KeOps symbolic variable, **solve** will
        assume that **self** defines an expression that is linear
        with respect to **var** and solve the equation ``self(var) = other``
        with respect to **var**.
      alpha (float, default=1e-10): Non-negative **ridge regularization** parameter.
      call (bool): If **True** and if no other symbolic variable than
        **var** is contained in **self**, **solve** will return a tensor
        solution of our linear system. Otherwise **solve** will return
        a callable :class:`LazyTensor`.
      backend (string): Specifies the map-reduce scheme,
        as detailed in the documentation of the :mod:`Genred <pykeops.torch.Genred>` module.
      device_id (int, default=-1): Specifies the GPU that should be used
        to perform the computation; a negative value lets your system
        choose the default GPU. This parameter is only useful if your
        system has access to several GPUs.
      ranges (6-uple of IntTensors, None by default):
        Ranges of integers that specify a
        :doc:`block-sparse reduction scheme <../../sparsity>`
        as detailed in the documentation of the :mod:`Genred <pykeops.torch.Genred>` module.
        If **None** (default), we simply use a **dense Kernel matrix**
        as we loop over all indices :math:`i\in[0,M)` and :math:`j\in[0,N)`.
      dtype_acc (string, default ``"auto"``): type for accumulator of reduction, before casting to dtype.
        It improves the accuracy of results in case of large sized data, but is slower.
        Default value "auto" will set this option to the value of dtype. The supported values are:
          - **dtype_acc** = ``"float16"`` : allowed only if dtype is "float16".
          - **dtype_acc** = ``"float32"`` : allowed only if dtype is "float16" or "float32".
          - **dtype_acc** = ``"float64"`` : allowed only if dtype is "float32" or "float64"..
      use_double_acc (bool, default False): same as setting dtype_acc="float64" (only one of the two options can be set)
        If True, accumulate results of reduction in float64 variables, before casting to float32.
        This can only be set to True when data is in float32 or float64.
        It improves the accuracy of results in case of large sized data, but is slower.
      sum_scheme (string, default ``"auto"``): method used to sum up results for reductions. This option may be changed only
        when reduction_op is one of: "Sum", "MaxSumShiftExp", "LogSumExp", "Max_SumShiftExpWeight", "LogSumExpWeight", "SumSoftMaxWeight".
        Default value "auto" will set this option to "block_red" for these reductions. Possible values are:
          - **sum_scheme** =  ``"direct_sum"``: direct summation
          - **sum_scheme** =  ``"block_sum"``: use an intermediate accumulator in each block before accumulating
            in the output. This improves accuracy for large sized data.
          - **sum_scheme** =  ``"kahan_scheme"``: use Kahan summation algorithm to compensate for round-off errors. This improves
            accuracy for large sized data.
        enable_chunks (bool, default True): enable automatic selection of special "chunked" computation mode for accelerating reductions
                            with formulas involving large dimension variables.

    .. warning::

        Please note that **no check** of symmetry and definiteness will be
        performed prior to our conjugate gradient descent.
    """

    if not hasattr(other, "__GenericLazyTensor__"):
        other = tensor.lt_constructor(
            x=other, axis=0
        )  # a vector is normally indexed by "i"

    # If given, var is symbolic variable corresponding to unknown
    # other must be a variable equal to the second member of the linear system,
    # and it may be symbolic. If it is symbolic, its index should match the index of var
    # if other is not symbolic, all variables in self must be non symbolic
    if len(other.symbolic_variables) == 0 and len(tensor.symbolic_variables) != 0:
        raise ValueError("If 'self' has symbolic variables, so should 'other'.")

    # we infer axis of reduction as the opposite of the axis of output
    axis = 1 - other.axis

    if var is None:
        # this is the classical mode: we want to invert sum(self*var) = other
        # we define var as a new symbolic variable with same dimension as other
        # and we assume axis of var is same as axis of reduction
        varindex = tensor.new_variable_index()
        var = tensor.lt_constructor((varindex, other.ndim, axis))
        res = tensor * var
    else:
        # var is given and must be a symbolic variable which is already inside self
        varindex = var.symbolic_variables[0][0]
        res = tensor.init()
        res.formula = tensor.formula

    res.formula2 = None
    res.reduction_op = "Solve"
    res.varindex = varindex
    res.varformula = var.formula.replace("VarSymb", "Var")
    res.other = other
    res.axis = axis

    kwargs_init, res.kwargs = tensor.separate_kwargs(kwargs)

    res.ndim = tensor.ndim

    if other.ndim > 100:
        res.rec_multVar_highdim = varindex
    else:
        res.rec_multVar_highdim = None

    from gempy_engine.modules.solver._pykeops_solvers._kernel_solver_from_lazy_tensor import KernelSolve
    if res._dtype is not None:
        res.fixvariables()
        res.callfun = KernelSolve(
            res.formula,
            [],
            res.varformula,
            res.axis,
            **kwargs_init,
            rec_multVar_highdim=res.rec_multVar_highdim,
        )

    # we call if call=True, if other is not symbolic, and if the dtype is set
    if call and len(other.symbolic_variables) == 0 and res._dtype is not None:
        return res()
    else:
        return res