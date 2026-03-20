Ah, that changes things significantly. You are completely right. If the number of basis functions, control points, or weights ($N$) varies per field (e.g., Field 1 has $N_1$ weights, Field 2 has $N_2$ weights), you cannot simply stack them into a neat $(N, F)$ matrix.PyKeOps requires uniform tensor dimensions for standard operations. However, you can still evaluate all of these differently shaped fields in a single PyKeOps operation using Block-Sparse Reductions.Here is how you handle ragged/differently shaped fields in PyKeOps without losing the single-operation performance.The Solution: Block-Diagonal Slices (ranges)Instead of trying to stack arrays of different sizes, you concatenate them into massive 1D arrays and tell PyKeOps to only compute interactions within specific blocks.If you have $F$ fields, you concatenate all your target points ($M_1 + M_2 + \dots$), all your source points/centers ($N_1 + N_2 + \dots$), and all your weights. Then, you pass a ranges tuple to PyKeOps. This instructs the GPU to act like a block-diagonal matrix, skipping all the cross-field computations (e.g., it ensures the evaluation points for Field 1 are only multiplied by the weights of Field 1).How to Implement ItHere is the conceptual workflow of how you adapt your function for block-sparsity:1. Concatenate your inputs before the PyKeOps call:Instead of passing a list of different arrays, flatten everything into single contiguous arrays.Weights: Combine into one array of size $\sum N_k$.Coordinates (XYZ): Combine target points into size $\sum M_k$ and source centers into $\sum N_k$.2. Generate the ranges tuple:PyKeOps needs to know where each field starts and ends in those concatenated arrays. You build a ranges tuple that defines the block-diagonal structure.Pythonimport numpy as np

# Let's say you have 3 fields.
# M_sizes: number of evaluation points (i) per field.
# N_sizes: number of weights/centers (j) per field.
M_sizes = [1000, 1500, 800]  
N_sizes = [50, 120, 30]

# 1. Calculate the starting indices for each block
keep_i = np.cumsum([0] + M_sizes)
keep_j = np.cumsum([0] + N_sizes)

# 2. Build the ranges tuple for a block-diagonal structure
ranges_i = np.array([[keep_i[k], keep_i[k+1]] for k in range(len(M_sizes))])
slices_i = np.arange(len(M_sizes))
ranges_j = np.array([[keep_j[k], keep_j[k+1]] for k in range(len(N_sizes))])
slices_j = np.arange(len(N_sizes))

# PyKeOps expects a 6-tuple for ranges:
# (ranges_i, slices_i, ranges_j, slices_j, ranges_j, slices_j)
# (The last two are for parameter variables, usually identical to j)
ranges = (ranges_i, slices_i, ranges_j, slices_j, ranges_j, slices_j)
3. Inject the ranges into the evaluation:Once you have concatenated your arrays and built your ranges tuple, you pass it to your LazyTensor or the .sum() reduction.Python# Assuming lazy_weights and eval_kernel are now built from the CONCATENATED arrays
   scalar_fields_concat = (eval_kernel * lazy_weights).sum(
   axis=0,
   backend=backend_string,
   ranges=ranges  # <--- This is the magic keyword
   )

# scalar_fields_concat is now a 1D array of size sum(M_k).
# You can split it back into your individual fields using np.split:
split_indices = np.cumsum(M_sizes)[:-1]
individual_fields = np.split(scalar_fields_concat, split_indices)
Alternative: Zero-Padding (The "Brute Force" Way)If dealing with the ranges tuple feels too complex to integrate into yield_evaluation_kernel, and the size differences between your fields are relatively small (e.g., $N_1 = 45$, $N_2 = 50$), there is a lazier alternative:Find the maximum size $N_{max}$.Pad the weights of the smaller fields with 0.0 up to $N_{max}$.Pad their coordinates/centers with dummy values (e.g., 0.0).Run the standard $(N_{max}, F)$ broadcasting I showed in the previous answer.Because the padded weights are exactly 0.0, their mathematical contribution to the sum is zero.Pros: Much easier to code; requires no changes to PyKeOps logic.Cons: Wastes GPU cycles computing dummy kernel interactions. If Field 1 has 10 points and Field 2 has 10,000 points, padding Field 1 to 10,000 is wildly inefficient and you absolutely must use the ranges approach above instead.Would you like me to show you how to securely adapt your yield_evaluation_kernel to accept and pass the ranges argument down to the PyKeOps engine?