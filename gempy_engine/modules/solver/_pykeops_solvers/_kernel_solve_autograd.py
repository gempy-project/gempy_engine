import torch
from pykeops import default_device_id
from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import keops_binder
from pykeops.common.parse_type import get_type
from pykeops.torch.generic.generic_red import GenredAutograd

from ._conjugate_gradient import ConjugateGradientSolver
from ._nystrom import create_adaptive_nystrom_preconditioner


class KernelSolveAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """

    @staticmethod
    def forward(
            ctx,
            formula,
            aliases,
            varinvpos,
            alpha,
            backend,
            dtype,
            device_id_request,
            eps,
            ranges,
            optional_flags,
            rec_multVar_highdim,
            nx,
            ny,
            x0,
            *args
    ):
        # N.B. when rec_multVar_highdim option is set, it means that formula is of the form "sum(F*b)", where b is a variable
        # with large dimension. In this case we set option multVar_highdim to allow for the use of the special "final chunk" computation
        # mode. However, this may not be also true for the gradients of the same formula. In fact only the gradient
        # with respect to variable b will have the same form. Hence, we save optional_flags current status into ctx,
        # before adding the multVar_highdim option.
        ctx.optional_flags = optional_flags.copy()
        if rec_multVar_highdim:
            optional_flags["multVar_highdim"] = 1
        else:
            optional_flags["multVar_highdim"] = 0

        tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

        # number of batch dimensions
        # N.B. we assume here that there is at least a cat=0 or cat=1 variable in the formula...
        nbatchdims = max(len(arg.shape) for arg in args) - 2
        use_ranges = nbatchdims > 0 or ranges

        device_args = args[0].device
        if tagCPUGPU == 1 & tagHostDevice == 1:
            for i in range(1, len(args)):
                if args[i].device.index != device_args.index:
                    raise ValueError(
                        "[KeOps] Input arrays must be all located on the same device."
                    )

        if device_id_request == -1:  # -1 means auto setting
            if device_args.index:  # means args are on Gpu
                device_id_request = device_args.index
            else:
                device_id_request = default_device_id if tagCPUGPU == 1 else -1
        else:
            if device_args.index:
                if device_args.index != device_id_request:
                    raise ValueError(
                        "[KeOps] Gpu device id of arrays is different from device id requested for computation."
                    )

        myconv = keops_binder["nvrtc" if tagCPUGPU else "cpp"](
            tagCPUGPU,
            tag1D2D,
            tagHostDevice,
            use_ranges,
            device_id_request,
            formula,
            aliases,
            len(args),
            dtype,
            "torch",
            optional_flags,
        ).import_module()

        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.varinvpos = varinvpos
        ctx.alpha = alpha
        ctx.backend = backend
        ctx.dtype = dtype
        ctx.device_id_request = device_id_request
        ctx.eps = eps
        ctx.nx = nx
        ctx.ny = ny
        ctx.myconv = myconv
        ctx.ranges = ranges
        ctx.rec_multVar_highdim = rec_multVar_highdim
        ctx.optional_flags = optional_flags

        varinv = args[varinvpos]
        ctx.varinvpos = varinvpos

        def linop(var):
            newargs = args[:varinvpos] + (var,) + args[varinvpos + 1:]
            res = myconv.genred_pytorch(
                device_args, ranges, nx, ny, nbatchdims, None, *newargs
            )
            if alpha:
                res += alpha * var
            return res

        global copy
        if False: # * This does not work for gpu and so far it seems not to be specially better than direct solvers
            preconditioner = create_adaptive_nystrom_preconditioner(
                binding="torch",
                linop=linop,
                x_sample=varinv.data,
                strategy="conservative",
            )
        else:
            preconditioner = None
        
        result = ConjugateGradientSolver(
            binding="torch",
            linop=linop,
            b=varinv.data,
            eps=1e-4,
            x0=x0,
            regularization=None,
            preconditioning=preconditioner,
            adaptive_tolerance=False,
            max_iterations=500,
            verbose=False
        )

        # relying on the 'ctx.saved_variables' attribute is necessary  if you want to be able to differentiate the output
        #  of the backward once again. It helps pytorch to keep track of 'who is who'.
        ctx.save_for_backward(*args, result)

        return result

    @staticmethod
    def backward(ctx, G):
        formula = ctx.formula
        aliases = ctx.aliases
        varinvpos = ctx.varinvpos
        backend = ctx.backend
        alpha = ctx.alpha
        dtype = ctx.dtype
        device_id_request = ctx.device_id_request
        eps = ctx.eps
        nx = ctx.nx
        ny = ctx.ny
        myconv = ctx.myconv
        ranges = ctx.ranges
        optional_flags = ctx.optional_flags
        rec_multVar_highdim = ctx.rec_multVar_highdim

        args = ctx.saved_tensors[:-1]  # Unwrap the saved variables
        nargs = len(args)
        result = ctx.saved_tensors[-1]

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = (
                "Var("
                + str(nargs)
                + ","
                + str(myconv.dimout)
                + ","
                + str(myconv.tagIJ)
                + ")"
        )

        # there is also a new variable for the formula's output
        resvar = (
                "Var("
                + str(nargs + 1)
                + ","
                + str(myconv.dimout)
                + ","
                + str(myconv.tagIJ)
                + ")"
        )

        newargs = args[:varinvpos] + (G,) + args[varinvpos + 1:]
        KinvG = KernelSolveAutograd.apply(
            formula,
            aliases,
            varinvpos,
            alpha,
            backend,
            dtype,
            device_id_request,
            eps,
            ranges,
            optional_flags,
            rec_multVar_highdim,
            nx,
            ny,
            *newargs
        )

        grads = []  # list of gradients wrt. args;

        for var_ind, sig in enumerate(aliases):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[
                var_ind + 13
            ]:  # because of (formula, aliases, varinvpos, alpha, backend, dtype, device_id, eps, ranges, optional_flags, rec_multVar_highdim, nx, ny)
                grads.append(None)  # Don't waste time computing it.

            else:  # Otherwise, the current gradient is really needed by the user:
                if var_ind == varinvpos:
                    grads.append(KinvG)
                else:
                    # adding new aliases is way too dangerous if we want to compute
                    # second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
                    # New here (Joan) : we still add the new variables to the list of "aliases" (without giving new aliases for them)
                    # these will not be used in the C++ code,
                    # but are useful to keep track of the actual variables used in the formula
                    _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
                    var = "Var(" + str(pos) + "," + str(dim) + "," + str(cat) + ")"  # V
                    formula_g = (
                            "Grad_WithSavedForward("
                            + formula
                            + ", "
                            + var
                            + ", "
                            + eta
                            + ", "
                            + resvar
                            + ")"
                    )  # Grad<F,V,G,R>
                    aliases_g = aliases + [eta, resvar]
                    args_g = (
                            args[:varinvpos]
                            + (result,)
                            + args[varinvpos + 1:]
                            + (-KinvG,)
                            + (result,)
                    )  # Don't forget the gradient to backprop !

                    # N.B.: if I understand PyTorch's doc, we should redefine this function every time we use it?
                    genconv = GenredAutograd.apply

                    if (
                            cat == 2
                    ):  # we're referring to a parameter, so we'll have to sum both wrt 'i' and 'j'
                        # WARNING !! : here we rely on the implementation of DiffT in files in folder keopscore/core/formulas/reductions
                        # if tagI==cat of V is 2, then reduction is done wrt j, so we need to further sum output wrt i
                        grad = genconv(
                            formula_g,
                            aliases_g,
                            backend,
                            dtype,
                            device_id_request,
                            ranges,
                            optional_flags,
                            None,
                            nx,
                            ny,
                            None,
                            *args_g
                        )
                        # Then, sum 'grad' wrt 'i' :
                        # I think that '.sum''s backward introduces non-contiguous arrays,
                        # and is thus non-compatible with GenredAutograd: grad = grad.sum(0)
                        # We replace it with a 'handmade hack' :
                        grad = torch.ones(1, grad.shape[0]).type_as(grad.data) @ grad
                        grad = grad.view(-1)
                    else:
                        grad = genconv(
                            formula_g,
                            aliases_g,
                            backend,
                            dtype,
                            device_id_request,
                            ranges,
                            optional_flags,
                            None,
                            nx,
                            ny,
                            None,
                            *args_g
                        )
                    grads.append(grad)

        # Grads wrt. formula, aliases, varinvpos, alpha, backend, dtype, device_id_request, eps, ranges, optional_flags, rec_multVar_highdim, nx, ny, *args
        return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                *grads,
        )
