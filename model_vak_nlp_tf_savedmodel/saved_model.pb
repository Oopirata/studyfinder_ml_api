��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028��
�
sequential_1/dense_3/biasVarHandleOp*
_output_shapes
: **

debug_namesequential_1/dense_3/bias/*
dtype0*
shape:**
shared_namesequential_1/dense_3/bias
�
-sequential_1/dense_3/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_3/bias*
_output_shapes
:*
dtype0
�
sequential_1/dense_2/biasVarHandleOp*
_output_shapes
: **

debug_namesequential_1/dense_2/bias/*
dtype0*
shape:@**
shared_namesequential_1/dense_2/bias
�
-sequential_1/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_2/bias*
_output_shapes
:@*
dtype0
�
#sequential_1/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *4

debug_name&$sequential_1/embedding_1/embeddings/*
dtype0*
shape:	�'*4
shared_name%#sequential_1/embedding_1/embeddings
�
7sequential_1/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp#sequential_1/embedding_1/embeddings*
_output_shapes
:	�'*
dtype0
�
sequential_1/dense_2/kernelVarHandleOp*
_output_shapes
: *,

debug_namesequential_1/dense_2/kernel/*
dtype0*
shape
:@*,
shared_namesequential_1/dense_2/kernel
�
/sequential_1/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_2/kernel*
_output_shapes

:@*
dtype0
�
sequential_1/dense_3/kernelVarHandleOp*
_output_shapes
: *,

debug_namesequential_1/dense_3/kernel/*
dtype0*
shape
:@*,
shared_namesequential_1/dense_3/kernel
�
/sequential_1/dense_3/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_3/kernel*
_output_shapes

:@*
dtype0
�
sequential_1/dense_3/bias_1VarHandleOp*
_output_shapes
: *,

debug_namesequential_1/dense_3/bias_1/*
dtype0*
shape:*,
shared_namesequential_1/dense_3/bias_1
�
/sequential_1/dense_3/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/dense_3/bias_1*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpsequential_1/dense_3/bias_1*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
sequential_1/dense_3/kernel_1VarHandleOp*
_output_shapes
: *.

debug_name sequential_1/dense_3/kernel_1/*
dtype0*
shape
:@*.
shared_namesequential_1/dense_3/kernel_1
�
1sequential_1/dense_3/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/dense_3/kernel_1*
_output_shapes

:@*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpsequential_1/dense_3/kernel_1*
_class
loc:@Variable_1*
_output_shapes

:@*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:@*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:@*
dtype0
�
%seed_generator_1/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_1/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_1/seed_generator_state
�
9seed_generator_1/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_1/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_2/Initializer/ReadVariableOpReadVariableOp%seed_generator_1/seed_generator_state*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0	
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0	*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0	
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
�
sequential_1/dense_2/bias_1VarHandleOp*
_output_shapes
: *,

debug_namesequential_1/dense_2/bias_1/*
dtype0*
shape:@*,
shared_namesequential_1/dense_2/bias_1
�
/sequential_1/dense_2/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/dense_2/bias_1*
_output_shapes
:@*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpsequential_1/dense_2/bias_1*
_class
loc:@Variable_3*
_output_shapes
:@*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:@*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:@*
dtype0
�
sequential_1/dense_2/kernel_1VarHandleOp*
_output_shapes
: *.

debug_name sequential_1/dense_2/kernel_1/*
dtype0*
shape
:@*.
shared_namesequential_1/dense_2/kernel_1
�
1sequential_1/dense_2/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/dense_2/kernel_1*
_output_shapes

:@*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpsequential_1/dense_2/kernel_1*
_class
loc:@Variable_4*
_output_shapes

:@*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape
:@*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
i
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes

:@*
dtype0
�
%sequential_1/embedding_1/embeddings_1VarHandleOp*
_output_shapes
: *6

debug_name(&sequential_1/embedding_1/embeddings_1/*
dtype0*
shape:	�'*6
shared_name'%sequential_1/embedding_1/embeddings_1
�
9sequential_1/embedding_1/embeddings_1/Read/ReadVariableOpReadVariableOp%sequential_1/embedding_1/embeddings_1*
_output_shapes
:	�'*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp%sequential_1/embedding_1/embeddings_1*
_class
loc:@Variable_5*
_output_shapes
:	�'*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:	�'*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
j
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:	�'*
dtype0
w
serve_keras_tensor_8Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserve_keras_tensor_8%sequential_1/embedding_1/embeddings_1sequential_1/dense_2/kernel_1sequential_1/dense_2/bias_1sequential_1/dense_3/kernel_1sequential_1/dense_3/bias_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*7
config_proto'%

CPU

GPU2*0,1J 8� �J *5
f0R.
,__inference_signature_wrapper___call___11632
�
serving_default_keras_tensor_8Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_keras_tensor_8%sequential_1/embedding_1/embeddings_1sequential_1/dense_2/kernel_1sequential_1/dense_2/bias_1sequential_1/dense_3/kernel_1sequential_1/dense_3/bias_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*7
config_proto'%

CPU

GPU2*0,1J 8� �J *5
f0R.
,__inference_signature_wrapper___call___11647

NoOpNoOp
�

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�

value�
B�
 B�

�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
.
0
	1

2
3
4
5*
'
0
	1

2
3
4*

0*
'
0
1
2
3
4*
* 

trace_0* 
"
	serve
serving_default* 
JD
VARIABLE_VALUE
Variable_5&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_4&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_3&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_2&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_1&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEVariable&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEsequential_1/dense_3/kernel_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEsequential_1/dense_2/kernel_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE%sequential_1/embedding_1/embeddings_1+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEsequential_1/dense_2/bias_1+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEsequential_1/dense_3/bias_1+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variablesequential_1/dense_3/kernel_1sequential_1/dense_2/kernel_1%sequential_1/embedding_1/embeddings_1sequential_1/dense_2/bias_1sequential_1/dense_3/bias_1Const*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *7
config_proto'%

CPU

GPU2*0,1J 8� �J *'
f"R 
__inference__traced_save_11761
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variablesequential_1/dense_3/kernel_1sequential_1/dense_2/kernel_1%sequential_1/embedding_1/embeddings_1sequential_1/dense_2/bias_1sequential_1/dense_3/bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *7
config_proto'%

CPU

GPU2*0,1J 8� �J **
f%R#
!__inference__traced_restore_11803��
�3
�
__inference___call___11616
keras_tensor_8M
:sequential_1_1_embedding_1_1_shape_readvariableop_resource:	�'G
5sequential_1_1_dense_2_1_cast_readvariableop_resource:@F
8sequential_1_1_dense_2_1_biasadd_readvariableop_resource:@G
5sequential_1_1_dense_3_1_cast_readvariableop_resource:@F
8sequential_1_1_dense_3_1_biasadd_readvariableop_resource:
identity��/sequential_1_1/dense_2_1/BiasAdd/ReadVariableOp�,sequential_1_1/dense_2_1/Cast/ReadVariableOp�/sequential_1_1/dense_3_1/BiasAdd/ReadVariableOp�,sequential_1_1/dense_3_1/Cast/ReadVariableOp�4sequential_1_1/embedding_1_1/GatherV2/ReadVariableOpz
!sequential_1_1/embedding_1_1/CastCastkeras_tensor_8*

DstT0*

SrcT0*'
_output_shapes
:���������e
#sequential_1_1/embedding_1_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential_1_1/embedding_1_1/LessLess%sequential_1_1/embedding_1_1/Cast:y:0,sequential_1_1/embedding_1_1/Less/y:output:0*
T0*'
_output_shapes
:����������
1sequential_1_1/embedding_1_1/Shape/ReadVariableOpReadVariableOp:sequential_1_1_embedding_1_1_shape_readvariableop_resource*
_output_shapes
:	�'*
dtype0s
"sequential_1_1/embedding_1_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"�     z
0sequential_1_1/embedding_1_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_1_1/embedding_1_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_1_1/embedding_1_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_1_1/embedding_1_1/strided_sliceStridedSlice+sequential_1_1/embedding_1_1/Shape:output:09sequential_1_1/embedding_1_1/strided_slice/stack:output:0;sequential_1_1/embedding_1_1/strided_slice/stack_1:output:0;sequential_1_1/embedding_1_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 sequential_1_1/embedding_1_1/addAddV2%sequential_1_1/embedding_1_1/Cast:y:03sequential_1_1/embedding_1_1/strided_slice:output:0*
T0*'
_output_shapes
:����������
%sequential_1_1/embedding_1_1/SelectV2SelectV2%sequential_1_1/embedding_1_1/Less:z:0$sequential_1_1/embedding_1_1/add:z:0%sequential_1_1/embedding_1_1/Cast:y:0*
T0*'
_output_shapes
:����������
4sequential_1_1/embedding_1_1/GatherV2/ReadVariableOpReadVariableOp:sequential_1_1_embedding_1_1_shape_readvariableop_resource*
_output_shapes
:	�'*
dtype0l
*sequential_1_1/embedding_1_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1_1/embedding_1_1/GatherV2GatherV2<sequential_1_1/embedding_1_1/GatherV2/ReadVariableOp:value:0.sequential_1_1/embedding_1_1/SelectV2:output:03sequential_1_1/embedding_1_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:����������
Bsequential_1_1/global_average_pooling1d_1_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
0sequential_1_1/global_average_pooling1d_1_1/MeanMean.sequential_1_1/embedding_1_1/GatherV2:output:0Ksequential_1_1/global_average_pooling1d_1_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
,sequential_1_1/dense_2_1/Cast/ReadVariableOpReadVariableOp5sequential_1_1_dense_2_1_cast_readvariableop_resource*
_output_shapes

:@*
dtype0�
sequential_1_1/dense_2_1/MatMulMatMul9sequential_1_1/global_average_pooling1d_1_1/Mean:output:04sequential_1_1/dense_2_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/sequential_1_1/dense_2_1/BiasAdd/ReadVariableOpReadVariableOp8sequential_1_1_dense_2_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 sequential_1_1/dense_2_1/BiasAddBiasAdd)sequential_1_1/dense_2_1/MatMul:product:07sequential_1_1/dense_2_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
sequential_1_1/dense_2_1/ReluRelu)sequential_1_1/dense_2_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
,sequential_1_1/dense_3_1/Cast/ReadVariableOpReadVariableOp5sequential_1_1_dense_3_1_cast_readvariableop_resource*
_output_shapes

:@*
dtype0�
sequential_1_1/dense_3_1/MatMulMatMul+sequential_1_1/dense_2_1/Relu:activations:04sequential_1_1/dense_3_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/sequential_1_1/dense_3_1/BiasAdd/ReadVariableOpReadVariableOp8sequential_1_1_dense_3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 sequential_1_1/dense_3_1/BiasAddBiasAdd)sequential_1_1/dense_3_1/MatMul:product:07sequential_1_1/dense_3_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 sequential_1_1/dense_3_1/SoftmaxSoftmax)sequential_1_1/dense_3_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*sequential_1_1/dense_3_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^sequential_1_1/dense_2_1/BiasAdd/ReadVariableOp-^sequential_1_1/dense_2_1/Cast/ReadVariableOp0^sequential_1_1/dense_3_1/BiasAdd/ReadVariableOp-^sequential_1_1/dense_3_1/Cast/ReadVariableOp5^sequential_1_1/embedding_1_1/GatherV2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2b
/sequential_1_1/dense_2_1/BiasAdd/ReadVariableOp/sequential_1_1/dense_2_1/BiasAdd/ReadVariableOp2\
,sequential_1_1/dense_2_1/Cast/ReadVariableOp,sequential_1_1/dense_2_1/Cast/ReadVariableOp2b
/sequential_1_1/dense_3_1/BiasAdd/ReadVariableOp/sequential_1_1/dense_3_1/BiasAdd/ReadVariableOp2\
,sequential_1_1/dense_3_1/Cast/ReadVariableOp,sequential_1_1/dense_3_1/Cast/ReadVariableOp2l
4sequential_1_1/embedding_1_1/GatherV2/ReadVariableOp4sequential_1_1/embedding_1_1/GatherV2/ReadVariableOp:W S
'
_output_shapes
:���������
(
_user_specified_namekeras_tensor_8:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
,__inference_signature_wrapper___call___11647
keras_tensor_8
unknown:	�'
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallkeras_tensor_8unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*7
config_proto'%

CPU

GPU2*0,1J 8� �J *#
fR
__inference___call___11616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namekeras_tensor_8:%!

_user_specified_name11635:%!

_user_specified_name11637:%!

_user_specified_name11639:%!

_user_specified_name11641:%!

_user_specified_name11643
�[
�

__inference__traced_save_11761
file_prefix4
!read_disablecopyonread_variable_5:	�'5
#read_1_disablecopyonread_variable_4:@1
#read_2_disablecopyonread_variable_3:@1
#read_3_disablecopyonread_variable_2:	5
#read_4_disablecopyonread_variable_1:@/
!read_5_disablecopyonread_variable:H
6read_6_disablecopyonread_sequential_1_dense_3_kernel_1:@H
6read_7_disablecopyonread_sequential_1_dense_2_kernel_1:@Q
>read_8_disablecopyonread_sequential_1_embedding_1_embeddings_1:	�'B
4read_9_disablecopyonread_sequential_1_dense_2_bias_1:@C
5read_10_disablecopyonread_sequential_1_dense_3_bias_1:
savev2_const
identity_23��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_5*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_5^Read/DisableCopyOnRead*
_output_shapes
:	�'*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	�'b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�'h
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_variable_4*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_variable_4^Read_1/DisableCopyOnRead*
_output_shapes

:@*
dtype0^

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:@h
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_variable_3*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_variable_3^Read_2/DisableCopyOnRead*
_output_shapes
:@*
dtype0Z

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:@h
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_variable_2*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_variable_2^Read_3/DisableCopyOnRead*
_output_shapes
:*
dtype0	Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0	*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0	*
_output_shapes
:h
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_1*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_1^Read_4/DisableCopyOnRead*
_output_shapes

:@*
dtype0^

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes

:@c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:@f
Read_5/DisableCopyOnReadDisableCopyOnRead!read_5_disablecopyonread_variable*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp!read_5_disablecopyonread_variable^Read_5/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_6/DisableCopyOnReadDisableCopyOnRead6read_6_disablecopyonread_sequential_1_dense_3_kernel_1*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp6read_6_disablecopyonread_sequential_1_dense_3_kernel_1^Read_6/DisableCopyOnRead*
_output_shapes

:@*
dtype0_
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@{
Read_7/DisableCopyOnReadDisableCopyOnRead6read_7_disablecopyonread_sequential_1_dense_2_kernel_1*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp6read_7_disablecopyonread_sequential_1_dense_2_kernel_1^Read_7/DisableCopyOnRead*
_output_shapes

:@*
dtype0_
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_8/DisableCopyOnReadDisableCopyOnRead>read_8_disablecopyonread_sequential_1_embedding_1_embeddings_1*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp>read_8_disablecopyonread_sequential_1_embedding_1_embeddings_1^Read_8/DisableCopyOnRead*
_output_shapes
:	�'*
dtype0`
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�'f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�'y
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_sequential_1_dense_2_bias_1*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_sequential_1_dense_2_bias_1^Read_9/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_10/DisableCopyOnReadDisableCopyOnRead5read_10_disablecopyonread_sequential_1_dense_3_bias_1*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp5read_10_disablecopyonread_sequential_1_dense_3_bias_1^Read_10/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_22Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_23IdentityIdentity_22:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_1:($
"
_user_specified_name
Variable:=9
7
_user_specified_namesequential_1/dense_3/kernel_1:=9
7
_user_specified_namesequential_1/dense_2/kernel_1:E	A
?
_user_specified_name'%sequential_1/embedding_1/embeddings_1:;
7
5
_user_specified_namesequential_1/dense_2/bias_1:;7
5
_user_specified_namesequential_1/dense_3/bias_1:=9

_output_shapes
: 

_user_specified_nameConst
�	
�
,__inference_signature_wrapper___call___11632
keras_tensor_8
unknown:	�'
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallkeras_tensor_8unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*7
config_proto'%

CPU

GPU2*0,1J 8� �J *#
fR
__inference___call___11616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namekeras_tensor_8:%!

_user_specified_name11620:%!

_user_specified_name11622:%!

_user_specified_name11624:%!

_user_specified_name11626:%!

_user_specified_name11628
�7
�
!__inference__traced_restore_11803
file_prefix.
assignvariableop_variable_5:	�'/
assignvariableop_1_variable_4:@+
assignvariableop_2_variable_3:@+
assignvariableop_3_variable_2:	/
assignvariableop_4_variable_1:@)
assignvariableop_5_variable:B
0assignvariableop_6_sequential_1_dense_3_kernel_1:@B
0assignvariableop_7_sequential_1_dense_2_kernel_1:@K
8assignvariableop_8_sequential_1_embedding_1_embeddings_1:	�'<
.assignvariableop_9_sequential_1_dense_2_bias_1:@=
/assignvariableop_10_sequential_1_dense_3_bias_1:
identity_12��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_5Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_4Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_3Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_2Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_1Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variableIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp0assignvariableop_6_sequential_1_dense_3_kernel_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp0assignvariableop_7_sequential_1_dense_2_kernel_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_sequential_1_embedding_1_embeddings_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_sequential_1_dense_2_bias_1Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_sequential_1_dense_3_bias_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_12Identity_12:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_1:($
"
_user_specified_name
Variable:=9
7
_user_specified_namesequential_1/dense_3/kernel_1:=9
7
_user_specified_namesequential_1/dense_2/kernel_1:E	A
?
_user_specified_name'%sequential_1/embedding_1/embeddings_1:;
7
5
_user_specified_namesequential_1/dense_2/bias_1:;7
5
_user_specified_namesequential_1/dense_3/bias_1"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
?
keras_tensor_8-
serve_keras_tensor_8:0���������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict*�
serving_default�
I
keras_tensor_87
 serving_default_keras_tensor_8:0���������>
output_02
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
J
0
	1

2
3
4
5"
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
'
0"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trace_02�
__inference___call___11616�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *-�*
(�%
keras_tensor_8���������ztrace_0
7
	serve
serving_default"
signature_map
6:4	�'2#sequential_1/embedding_1/embeddings
-:+@2sequential_1/dense_2/kernel
':%@2sequential_1/dense_2/bias
1:/	2%seed_generator_1/seed_generator_state
-:+@2sequential_1/dense_3/kernel
':%2sequential_1/dense_3/bias
-:+@2sequential_1/dense_3/kernel
-:+@2sequential_1/dense_2/kernel
6:4	�'2#sequential_1/embedding_1/embeddings
':%@2sequential_1/dense_2/bias
':%2sequential_1/dense_3/bias
�B�
__inference___call___11616keras_tensor_8"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_signature_wrapper___call___11632keras_tensor_8"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 #

kwonlyargs�
jkeras_tensor_8
kwonlydefaults
 
annotations� *
 
�B�
,__inference_signature_wrapper___call___11647keras_tensor_8"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 #

kwonlyargs�
jkeras_tensor_8
kwonlydefaults
 
annotations� *
 �
__inference___call___11616c	
7�4
-�*
(�%
keras_tensor_8���������
� "!�
unknown����������
,__inference_signature_wrapper___call___11632�	
I�F
� 
?�<
:
keras_tensor_8(�%
keras_tensor_8���������"3�0
.
output_0"�
output_0����������
,__inference_signature_wrapper___call___11647�	
I�F
� 
?�<
:
keras_tensor_8(�%
keras_tensor_8���������"3�0
.
output_0"�
output_0���������