ѕЌ
ЭЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18нг
Ђ
"l6_integration/l4_integ_fc0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ў*3
shared_name$"l6_integration/l4_integ_fc0/kernel

6l6_integration/l4_integ_fc0/kernel/Read/ReadVariableOpReadVariableOp"l6_integration/l4_integ_fc0/kernel* 
_output_shapes
:
ў*
dtype0

 l6_integration/l4_integ_fc0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" l6_integration/l4_integ_fc0/bias

4l6_integration/l4_integ_fc0/bias/Read/ReadVariableOpReadVariableOp l6_integration/l4_integ_fc0/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
Р

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ћ	
valueё	Bю	 Bч	

layer_with_weights-0
layer-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
|
_inbound_nodes

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
f
_inbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
 

	0

1

	0

1
­
regularization_losses
metrics

layers
non_trainable_variables
trainable_variables
layer_metrics
layer_regularization_losses
	variables
 
 
nl
VARIABLE_VALUE"l6_integration/l4_integ_fc0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE l6_integration/l4_integ_fc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
­
regularization_losses
layer_metrics
metrics
non_trainable_variables
trainable_variables

layers
layer_regularization_losses
	variables
 
 
 
 
­
regularization_losses
layer_metrics
metrics
 non_trainable_variables
trainable_variables

!layers
"layer_regularization_losses
	variables
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 

"serving_default_l4_integ_fc0_inputPlaceholder*(
_output_shapes
:џџџџџџџџџў*
dtype0*
shape:џџџџџџџџџў

StatefulPartitionedCallStatefulPartitionedCall"serving_default_l4_integ_fc0_input"l6_integration/l4_integ_fc0/kernel l6_integration/l4_integ_fc0/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_48108
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6l6_integration/l4_integ_fc0/kernel/Read/ReadVariableOp4l6_integration/l4_integ_fc0/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_48206
н
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"l6_integration/l4_integ_fc0/kernel l6_integration/l4_integ_fc0/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_48222ЛЙ
Ь
я
__inference__traced_save_48206
file_prefixA
=savev2_l6_integration_l4_integ_fc0_kernel_read_readvariableop?
;savev2_l6_integration_l4_integ_fc0_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_713f5fe81dcb48ae8a0eb6d99ce7a5c7/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ё
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slicesИ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_l6_integration_l4_integ_fc0_kernel_read_readvariableop;savev2_l6_integration_l4_integ_fc0_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
: :
ў:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ў:!

_output_shapes	
::

_output_shapes
: 
и
Џ
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_48018

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ў*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў:::P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs
А
d
H__inference_activation_21_layer_call_and_return_conditional_losses_48172

inputs
identityO
TanhTanhinputs*
T0*(
_output_shapes
:џџџџџџџџџ2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б
Ы
I__inference_l6_integration_layer_call_and_return_conditional_losses_48058
l4_integ_fc0_input
l4_integ_fc0_48051
l4_integ_fc0_48053
identityЂ$l4_integ_fc0/StatefulPartitionedCallВ
$l4_integ_fc0/StatefulPartitionedCallStatefulPartitionedCalll4_integ_fc0_inputl4_integ_fc0_48051l4_integ_fc0_48053*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_480182&
$l4_integ_fc0/StatefulPartitionedCall
activation_21/PartitionedCallPartitionedCall-l4_integ_fc0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_480392
activation_21/PartitionedCallЂ
IdentityIdentity&activation_21/PartitionedCall:output:0%^l4_integ_fc0/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::2L
$l4_integ_fc0/StatefulPartitionedCall$l4_integ_fc0/StatefulPartitionedCall:\ X
(
_output_shapes
:џџџџџџџџџў
,
_user_specified_namel4_integ_fc0_input

П
I__inference_l6_integration_layer_call_and_return_conditional_losses_48090

inputs
l4_integ_fc0_48083
l4_integ_fc0_48085
identityЂ$l4_integ_fc0/StatefulPartitionedCallІ
$l4_integ_fc0/StatefulPartitionedCallStatefulPartitionedCallinputsl4_integ_fc0_48083l4_integ_fc0_48085*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_480182&
$l4_integ_fc0/StatefulPartitionedCall
activation_21/PartitionedCallPartitionedCall-l4_integ_fc0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_480392
activation_21/PartitionedCallЂ
IdentityIdentity&activation_21/PartitionedCall:output:0%^l4_integ_fc0/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::2L
$l4_integ_fc0/StatefulPartitionedCall$l4_integ_fc0/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs
ч

,__inference_l4_integ_fc0_layer_call_fn_48167

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_480182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs


.__inference_l6_integration_layer_call_fn_48078
l4_integ_fc0_input
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalll4_integ_fc0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_l6_integration_layer_call_and_return_conditional_losses_480712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:џџџџџџџџџў
,
_user_specified_namel4_integ_fc0_input
А
d
H__inference_activation_21_layer_call_and_return_conditional_losses_48039

inputs
identityO
TanhTanhinputs*
T0*(
_output_shapes
:џџџџџџџџџ2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б
Ы
I__inference_l6_integration_layer_call_and_return_conditional_losses_48048
l4_integ_fc0_input
l4_integ_fc0_48029
l4_integ_fc0_48031
identityЂ$l4_integ_fc0/StatefulPartitionedCallВ
$l4_integ_fc0/StatefulPartitionedCallStatefulPartitionedCalll4_integ_fc0_inputl4_integ_fc0_48029l4_integ_fc0_48031*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_480182&
$l4_integ_fc0/StatefulPartitionedCall
activation_21/PartitionedCallPartitionedCall-l4_integ_fc0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_480392
activation_21/PartitionedCallЂ
IdentityIdentity&activation_21/PartitionedCall:output:0%^l4_integ_fc0/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::2L
$l4_integ_fc0/StatefulPartitionedCall$l4_integ_fc0/StatefulPartitionedCall:\ X
(
_output_shapes
:џџџџџџџџџў
,
_user_specified_namel4_integ_fc0_input
Р
с
!__inference__traced_restore_48222
file_prefix7
3assignvariableop_l6_integration_l4_integ_fc0_kernel7
3assignvariableop_1_l6_integration_l4_integ_fc0_bias

identity_3ЂAssignVariableOpЂAssignVariableOp_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ё
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slicesК
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityВ
AssignVariableOpAssignVariableOp3assignvariableop_l6_integration_l4_integ_fc0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1И
AssignVariableOp_1AssignVariableOp3assignvariableop_1_l6_integration_l4_integ_fc0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2

Identity_3IdentityIdentity_2:output:0^AssignVariableOp^AssignVariableOp_1*
T0*
_output_shapes
: 2

Identity_3"!

identity_3Identity_3:output:0*
_input_shapes

: ::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

I
-__inference_activation_21_layer_call_fn_48177

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_480392
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и
Џ
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_48158

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ў*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў:::P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs
Љ

Ы
I__inference_l6_integration_layer_call_and_return_conditional_losses_48119

inputs/
+l4_integ_fc0_matmul_readvariableop_resource0
,l4_integ_fc0_biasadd_readvariableop_resource
identityЖ
"l4_integ_fc0/MatMul/ReadVariableOpReadVariableOp+l4_integ_fc0_matmul_readvariableop_resource* 
_output_shapes
:
ў*
dtype02$
"l4_integ_fc0/MatMul/ReadVariableOp
l4_integ_fc0/MatMulMatMulinputs*l4_integ_fc0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
l4_integ_fc0/MatMulД
#l4_integ_fc0/BiasAdd/ReadVariableOpReadVariableOp,l4_integ_fc0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#l4_integ_fc0/BiasAdd/ReadVariableOpЖ
l4_integ_fc0/BiasAddBiasAddl4_integ_fc0/MatMul:product:0+l4_integ_fc0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
l4_integ_fc0/BiasAdd
activation_21/TanhTanhl4_integ_fc0/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
activation_21/Tanhk
IdentityIdentityactivation_21/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў:::P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs
Љ

Ы
I__inference_l6_integration_layer_call_and_return_conditional_losses_48130

inputs/
+l4_integ_fc0_matmul_readvariableop_resource0
,l4_integ_fc0_biasadd_readvariableop_resource
identityЖ
"l4_integ_fc0/MatMul/ReadVariableOpReadVariableOp+l4_integ_fc0_matmul_readvariableop_resource* 
_output_shapes
:
ў*
dtype02$
"l4_integ_fc0/MatMul/ReadVariableOp
l4_integ_fc0/MatMulMatMulinputs*l4_integ_fc0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
l4_integ_fc0/MatMulД
#l4_integ_fc0/BiasAdd/ReadVariableOpReadVariableOp,l4_integ_fc0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#l4_integ_fc0/BiasAdd/ReadVariableOpЖ
l4_integ_fc0/BiasAddBiasAddl4_integ_fc0/MatMul:product:0+l4_integ_fc0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
l4_integ_fc0/BiasAdd
activation_21/TanhTanhl4_integ_fc0/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
activation_21/Tanhk
IdentityIdentityactivation_21/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў:::P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs


.__inference_l6_integration_layer_call_fn_48097
l4_integ_fc0_input
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalll4_integ_fc0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_l6_integration_layer_call_and_return_conditional_losses_480902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:џџџџџџџџџў
,
_user_specified_namel4_integ_fc0_input
ы

.__inference_l6_integration_layer_call_fn_48148

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_l6_integration_layer_call_and_return_conditional_losses_480902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs
л

#__inference_signature_wrapper_48108
l4_integ_fc0_input
unknown
	unknown_0
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCalll4_integ_fc0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_480042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:џџџџџџџџџў
,
_user_specified_namel4_integ_fc0_input
С
Ь
 __inference__wrapped_model_48004
l4_integ_fc0_input>
:l6_integration_l4_integ_fc0_matmul_readvariableop_resource?
;l6_integration_l4_integ_fc0_biasadd_readvariableop_resource
identityу
1l6_integration/l4_integ_fc0/MatMul/ReadVariableOpReadVariableOp:l6_integration_l4_integ_fc0_matmul_readvariableop_resource* 
_output_shapes
:
ў*
dtype023
1l6_integration/l4_integ_fc0/MatMul/ReadVariableOpд
"l6_integration/l4_integ_fc0/MatMulMatMull4_integ_fc0_input9l6_integration/l4_integ_fc0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2$
"l6_integration/l4_integ_fc0/MatMulс
2l6_integration/l4_integ_fc0/BiasAdd/ReadVariableOpReadVariableOp;l6_integration_l4_integ_fc0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2l6_integration/l4_integ_fc0/BiasAdd/ReadVariableOpђ
#l6_integration/l4_integ_fc0/BiasAddBiasAdd,l6_integration/l4_integ_fc0/MatMul:product:0:l6_integration/l4_integ_fc0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2%
#l6_integration/l4_integ_fc0/BiasAddЏ
!l6_integration/activation_21/TanhTanh,l6_integration/l4_integ_fc0/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2#
!l6_integration/activation_21/Tanhz
IdentityIdentity%l6_integration/activation_21/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў:::\ X
(
_output_shapes
:џџџџџџџџџў
,
_user_specified_namel4_integ_fc0_input

П
I__inference_l6_integration_layer_call_and_return_conditional_losses_48071

inputs
l4_integ_fc0_48064
l4_integ_fc0_48066
identityЂ$l4_integ_fc0/StatefulPartitionedCallІ
$l4_integ_fc0/StatefulPartitionedCallStatefulPartitionedCallinputsl4_integ_fc0_48064l4_integ_fc0_48066*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_480182&
$l4_integ_fc0/StatefulPartitionedCall
activation_21/PartitionedCallPartitionedCall-l4_integ_fc0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_480392
activation_21/PartitionedCallЂ
IdentityIdentity&activation_21/PartitionedCall:output:0%^l4_integ_fc0/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::2L
$l4_integ_fc0/StatefulPartitionedCall$l4_integ_fc0/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs
ы

.__inference_l6_integration_layer_call_fn_48139

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_l6_integration_layer_call_and_return_conditional_losses_480712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџў::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџў
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ш
serving_defaultД
R
l4_integ_fc0_input<
$serving_default_l4_integ_fc0_input:0џџџџџџџџџўB
activation_211
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ЧQ
і
layer_with_weights-0
layer-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*#&call_and_return_all_conditional_losses
$__call__
%_default_save_signature"
_tf_keras_sequentialч{"class_name": "Sequential", "name": "l6_integration", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "l6_integration", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 254]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "l4_integ_fc0_input"}}, {"class_name": "Dense", "config": {"name": "l4_integ_fc0", "trainable": true, "dtype": "float32", "units": 129, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "tanh"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 254}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 254]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "l6_integration", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 254]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "l4_integ_fc0_input"}}, {"class_name": "Dense", "config": {"name": "l4_integ_fc0", "trainable": true, "dtype": "float32", "units": 129, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "tanh"}}]}}}

_inbound_nodes

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
*&&call_and_return_all_conditional_losses
'__call__"к
_tf_keras_layerР{"class_name": "Dense", "name": "l4_integ_fc0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "l4_integ_fc0", "trainable": true, "dtype": "float32", "units": 129, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 254}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 254]}}
ы
_inbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
*(&call_and_return_all_conditional_losses
)__call__"Ш
_tf_keras_layerЎ{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "tanh"}}
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
Ъ
regularization_losses
metrics

layers
non_trainable_variables
trainable_variables
layer_metrics
layer_regularization_losses
	variables
$__call__
%_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
,
*serving_default"
signature_map
 "
trackable_list_wrapper
6:4
ў2"l6_integration/l4_integ_fc0/kernel
/:-2 l6_integration/l4_integ_fc0/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
regularization_losses
layer_metrics
metrics
non_trainable_variables
trainable_variables

layers
layer_regularization_losses
	variables
'__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
layer_metrics
metrics
 non_trainable_variables
trainable_variables

!layers
"layer_regularization_losses
	variables
)__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ђ2я
I__inference_l6_integration_layer_call_and_return_conditional_losses_48130
I__inference_l6_integration_layer_call_and_return_conditional_losses_48119
I__inference_l6_integration_layer_call_and_return_conditional_losses_48058
I__inference_l6_integration_layer_call_and_return_conditional_losses_48048Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
.__inference_l6_integration_layer_call_fn_48139
.__inference_l6_integration_layer_call_fn_48097
.__inference_l6_integration_layer_call_fn_48078
.__inference_l6_integration_layer_call_fn_48148Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
 __inference__wrapped_model_48004Т
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *2Ђ/
-*
l4_integ_fc0_inputџџџџџџџџџў
ё2ю
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_48158Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_l4_integ_fc0_layer_call_fn_48167Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_activation_21_layer_call_and_return_conditional_losses_48172Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_activation_21_layer_call_fn_48177Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
=B;
#__inference_signature_wrapper_48108l4_integ_fc0_inputЇ
 __inference__wrapped_model_48004	
<Ђ9
2Ђ/
-*
l4_integ_fc0_inputџџџџџџџџџў
Њ ">Њ;
9
activation_21(%
activation_21џџџџџџџџџІ
H__inference_activation_21_layer_call_and_return_conditional_losses_48172Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 ~
-__inference_activation_21_layer_call_fn_48177M0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЉ
G__inference_l4_integ_fc0_layer_call_and_return_conditional_losses_48158^	
0Ђ-
&Ђ#
!
inputsџџџџџџџџџў
Њ "&Ђ#

0џџџџџџџџџ
 
,__inference_l4_integ_fc0_layer_call_fn_48167Q	
0Ђ-
&Ђ#
!
inputsџџџџџџџџџў
Њ "џџџџџџџџџП
I__inference_l6_integration_layer_call_and_return_conditional_losses_48048r	
DЂA
:Ђ7
-*
l4_integ_fc0_inputџџџџџџџџџў
p

 
Њ "&Ђ#

0џџџџџџџџџ
 П
I__inference_l6_integration_layer_call_and_return_conditional_losses_48058r	
DЂA
:Ђ7
-*
l4_integ_fc0_inputџџџџџџџџџў
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Г
I__inference_l6_integration_layer_call_and_return_conditional_losses_48119f	
8Ђ5
.Ђ+
!
inputsџџџџџџџџџў
p

 
Њ "&Ђ#

0џџџџџџџџџ
 Г
I__inference_l6_integration_layer_call_and_return_conditional_losses_48130f	
8Ђ5
.Ђ+
!
inputsџџџџџџџџџў
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 
.__inference_l6_integration_layer_call_fn_48078e	
DЂA
:Ђ7
-*
l4_integ_fc0_inputџџџџџџџџџў
p

 
Њ "џџџџџџџџџ
.__inference_l6_integration_layer_call_fn_48097e	
DЂA
:Ђ7
-*
l4_integ_fc0_inputџџџџџџџџџў
p 

 
Њ "џџџџџџџџџ
.__inference_l6_integration_layer_call_fn_48139Y	
8Ђ5
.Ђ+
!
inputsџџџџџџџџџў
p

 
Њ "џџџџџџџџџ
.__inference_l6_integration_layer_call_fn_48148Y	
8Ђ5
.Ђ+
!
inputsџџџџџџџџџў
p 

 
Њ "џџџџџџџџџР
#__inference_signature_wrapper_48108	
RЂO
Ђ 
HЊE
C
l4_integ_fc0_input-*
l4_integ_fc0_inputџџџџџџџџџў">Њ;
9
activation_21(%
activation_21џџџџџџџџџ