??"
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
training_334/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *'
shared_nametraining_334/Adam/iter
y
*training_334/Adam/iter/Read/ReadVariableOpReadVariableOptraining_334/Adam/iter*
_output_shapes
: *
dtype0	
?
training_334/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_334/Adam/beta_1
}
,training_334/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_334/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_334/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_334/Adam/beta_2
}
,training_334/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_334/Adam/beta_2*
_output_shapes
: *
dtype0
?
training_334/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametraining_334/Adam/decay
{
+training_334/Adam/decay/Read/ReadVariableOpReadVariableOptraining_334/Adam/decay*
_output_shapes
: *
dtype0
?
training_334/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training_334/Adam/learning_rate
?
3training_334/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_334/Adam/learning_rate*
_output_shapes
: *
dtype0
}
dense_421/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *!
shared_namedense_421/kernel
v
$dense_421/kernel/Read/ReadVariableOpReadVariableOpdense_421/kernel*
_output_shapes
:	? *
dtype0
t
dense_421/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_421/bias
m
"dense_421/bias/Read/ReadVariableOpReadVariableOpdense_421/bias*
_output_shapes
: *
dtype0
}
dense_422/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*!
shared_namedense_422/kernel
v
$dense_422/kernel/Read/ReadVariableOpReadVariableOpdense_422/kernel*
_output_shapes
:	 ?*
dtype0
u
dense_422/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_422/bias
n
"dense_422/bias/Read/ReadVariableOpReadVariableOpdense_422/bias*
_output_shapes	
:?*
dtype0
~
dense_423/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_423/kernel
w
$dense_423/kernel/Read/ReadVariableOpReadVariableOpdense_423/kernel* 
_output_shapes
:
??*
dtype0
u
dense_423/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_423/bias
n
"dense_423/bias/Read/ReadVariableOpReadVariableOpdense_423/bias*
_output_shapes	
:?*
dtype0
~
dense_424/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_424/kernel
w
$dense_424/kernel/Read/ReadVariableOpReadVariableOpdense_424/kernel* 
_output_shapes
:
??*
dtype0
u
dense_424/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_424/bias
n
"dense_424/bias/Read/ReadVariableOpReadVariableOpdense_424/bias*
_output_shapes	
:?*
dtype0
~
dense_425/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_425/kernel
w
$dense_425/kernel/Read/ReadVariableOpReadVariableOpdense_425/kernel* 
_output_shapes
:
??*
dtype0
u
dense_425/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_425/bias
n
"dense_425/bias/Read/ReadVariableOpReadVariableOpdense_425/bias*
_output_shapes	
:?*
dtype0
~
dense_426/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_426/kernel
w
$dense_426/kernel/Read/ReadVariableOpReadVariableOpdense_426/kernel* 
_output_shapes
:
??*
dtype0
u
dense_426/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_426/bias
n
"dense_426/bias/Read/ReadVariableOpReadVariableOpdense_426/bias*
_output_shapes	
:?*
dtype0
}
dense_427/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_427/kernel
v
$dense_427/kernel/Read/ReadVariableOpReadVariableOpdense_427/kernel*
_output_shapes
:	?*
dtype0
t
dense_427/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_427/bias
m
"dense_427/bias/Read/ReadVariableOpReadVariableOpdense_427/bias*
_output_shapes
:*
dtype0
|
dense_428/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_428/kernel
u
$dense_428/kernel/Read/ReadVariableOpReadVariableOpdense_428/kernel*
_output_shapes

:*
dtype0
t
dense_428/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_428/bias
m
"dense_428/bias/Read/ReadVariableOpReadVariableOpdense_428/bias*
_output_shapes
:*
dtype0
?
training_320/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *'
shared_nametraining_320/Adam/iter
y
*training_320/Adam/iter/Read/ReadVariableOpReadVariableOptraining_320/Adam/iter*
_output_shapes
: *
dtype0	
?
training_320/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_320/Adam/beta_1
}
,training_320/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_320/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_320/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_320/Adam/beta_2
}
,training_320/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_320/Adam/beta_2*
_output_shapes
: *
dtype0
?
training_320/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametraining_320/Adam/decay
{
+training_320/Adam/decay/Read/ReadVariableOpReadVariableOptraining_320/Adam/decay*
_output_shapes
: *
dtype0
?
training_320/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training_320/Adam/learning_rate
?
3training_320/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_320/Adam/learning_rate*
_output_shapes
: *
dtype0
f
	total_346VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_346
_
total_346/Read/ReadVariableOpReadVariableOp	total_346*
_output_shapes
: *
dtype0
f
	count_346VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_346
_
count_346/Read/ReadVariableOpReadVariableOp	count_346*
_output_shapes
: *
dtype0
f
	total_330VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_330
_
total_330/Read/ReadVariableOpReadVariableOp	total_330*
_output_shapes
: *
dtype0
f
	count_330VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_330
_
count_330/Read/ReadVariableOpReadVariableOp	count_330*
_output_shapes
: *
dtype0
f
	total_331VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_331
_
total_331/Read/ReadVariableOpReadVariableOp	total_331*
_output_shapes
: *
dtype0
f
	count_331VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_331
_
count_331/Read/ReadVariableOpReadVariableOp	count_331*
_output_shapes
: *
dtype0
?
$training_334/Adam/dense_421/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *5
shared_name&$training_334/Adam/dense_421/kernel/m
?
8training_334/Adam/dense_421/kernel/m/Read/ReadVariableOpReadVariableOp$training_334/Adam/dense_421/kernel/m*
_output_shapes
:	? *
dtype0
?
"training_334/Adam/dense_421/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_334/Adam/dense_421/bias/m
?
6training_334/Adam/dense_421/bias/m/Read/ReadVariableOpReadVariableOp"training_334/Adam/dense_421/bias/m*
_output_shapes
: *
dtype0
?
$training_334/Adam/dense_422/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*5
shared_name&$training_334/Adam/dense_422/kernel/m
?
8training_334/Adam/dense_422/kernel/m/Read/ReadVariableOpReadVariableOp$training_334/Adam/dense_422/kernel/m*
_output_shapes
:	 ?*
dtype0
?
"training_334/Adam/dense_422/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_334/Adam/dense_422/bias/m
?
6training_334/Adam/dense_422/bias/m/Read/ReadVariableOpReadVariableOp"training_334/Adam/dense_422/bias/m*
_output_shapes	
:?*
dtype0
?
$training_334/Adam/dense_423/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_334/Adam/dense_423/kernel/m
?
8training_334/Adam/dense_423/kernel/m/Read/ReadVariableOpReadVariableOp$training_334/Adam/dense_423/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_334/Adam/dense_423/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_334/Adam/dense_423/bias/m
?
6training_334/Adam/dense_423/bias/m/Read/ReadVariableOpReadVariableOp"training_334/Adam/dense_423/bias/m*
_output_shapes	
:?*
dtype0
?
$training_334/Adam/dense_421/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *5
shared_name&$training_334/Adam/dense_421/kernel/v
?
8training_334/Adam/dense_421/kernel/v/Read/ReadVariableOpReadVariableOp$training_334/Adam/dense_421/kernel/v*
_output_shapes
:	? *
dtype0
?
"training_334/Adam/dense_421/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_334/Adam/dense_421/bias/v
?
6training_334/Adam/dense_421/bias/v/Read/ReadVariableOpReadVariableOp"training_334/Adam/dense_421/bias/v*
_output_shapes
: *
dtype0
?
$training_334/Adam/dense_422/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*5
shared_name&$training_334/Adam/dense_422/kernel/v
?
8training_334/Adam/dense_422/kernel/v/Read/ReadVariableOpReadVariableOp$training_334/Adam/dense_422/kernel/v*
_output_shapes
:	 ?*
dtype0
?
"training_334/Adam/dense_422/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_334/Adam/dense_422/bias/v
?
6training_334/Adam/dense_422/bias/v/Read/ReadVariableOpReadVariableOp"training_334/Adam/dense_422/bias/v*
_output_shapes	
:?*
dtype0
?
$training_334/Adam/dense_423/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_334/Adam/dense_423/kernel/v
?
8training_334/Adam/dense_423/kernel/v/Read/ReadVariableOpReadVariableOp$training_334/Adam/dense_423/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_334/Adam/dense_423/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_334/Adam/dense_423/bias/v
?
6training_334/Adam/dense_423/bias/v/Read/ReadVariableOpReadVariableOp"training_334/Adam/dense_423/bias/v*
_output_shapes	
:?*
dtype0
?
$training_320/Adam/dense_424/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_320/Adam/dense_424/kernel/m
?
8training_320/Adam/dense_424/kernel/m/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_424/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_320/Adam/dense_424/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_320/Adam/dense_424/bias/m
?
6training_320/Adam/dense_424/bias/m/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_424/bias/m*
_output_shapes	
:?*
dtype0
?
$training_320/Adam/dense_425/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_320/Adam/dense_425/kernel/m
?
8training_320/Adam/dense_425/kernel/m/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_425/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_320/Adam/dense_425/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_320/Adam/dense_425/bias/m
?
6training_320/Adam/dense_425/bias/m/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_425/bias/m*
_output_shapes	
:?*
dtype0
?
$training_320/Adam/dense_426/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_320/Adam/dense_426/kernel/m
?
8training_320/Adam/dense_426/kernel/m/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_426/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_320/Adam/dense_426/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_320/Adam/dense_426/bias/m
?
6training_320/Adam/dense_426/bias/m/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_426/bias/m*
_output_shapes	
:?*
dtype0
?
$training_320/Adam/dense_427/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*5
shared_name&$training_320/Adam/dense_427/kernel/m
?
8training_320/Adam/dense_427/kernel/m/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_427/kernel/m*
_output_shapes
:	?*
dtype0
?
"training_320/Adam/dense_427/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_320/Adam/dense_427/bias/m
?
6training_320/Adam/dense_427/bias/m/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_427/bias/m*
_output_shapes
:*
dtype0
?
$training_320/Adam/dense_428/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$training_320/Adam/dense_428/kernel/m
?
8training_320/Adam/dense_428/kernel/m/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_428/kernel/m*
_output_shapes

:*
dtype0
?
"training_320/Adam/dense_428/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_320/Adam/dense_428/bias/m
?
6training_320/Adam/dense_428/bias/m/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_428/bias/m*
_output_shapes
:*
dtype0
?
$training_320/Adam/dense_424/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_320/Adam/dense_424/kernel/v
?
8training_320/Adam/dense_424/kernel/v/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_424/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_320/Adam/dense_424/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_320/Adam/dense_424/bias/v
?
6training_320/Adam/dense_424/bias/v/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_424/bias/v*
_output_shapes	
:?*
dtype0
?
$training_320/Adam/dense_425/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_320/Adam/dense_425/kernel/v
?
8training_320/Adam/dense_425/kernel/v/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_425/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_320/Adam/dense_425/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_320/Adam/dense_425/bias/v
?
6training_320/Adam/dense_425/bias/v/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_425/bias/v*
_output_shapes	
:?*
dtype0
?
$training_320/Adam/dense_426/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_320/Adam/dense_426/kernel/v
?
8training_320/Adam/dense_426/kernel/v/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_426/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_320/Adam/dense_426/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_320/Adam/dense_426/bias/v
?
6training_320/Adam/dense_426/bias/v/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_426/bias/v*
_output_shapes	
:?*
dtype0
?
$training_320/Adam/dense_427/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*5
shared_name&$training_320/Adam/dense_427/kernel/v
?
8training_320/Adam/dense_427/kernel/v/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_427/kernel/v*
_output_shapes
:	?*
dtype0
?
"training_320/Adam/dense_427/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_320/Adam/dense_427/bias/v
?
6training_320/Adam/dense_427/bias/v/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_427/bias/v*
_output_shapes
:*
dtype0
?
$training_320/Adam/dense_428/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$training_320/Adam/dense_428/kernel/v
?
8training_320/Adam/dense_428/kernel/v/Read/ReadVariableOpReadVariableOp$training_320/Adam/dense_428/kernel/v*
_output_shapes

:*
dtype0
?
"training_320/Adam/dense_428/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_320/Adam/dense_428/bias/v
?
6training_320/Adam/dense_428/bias/v/Read/ReadVariableOpReadVariableOp"training_320/Adam/dense_428/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?l
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?k
value?kB?k B?k
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
 
?
layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
?
 iter

!beta_1

"beta_2
	#decay
$learning_rate%m?&m?'m?(m?)m?*m?%v?&v?'v?(v?)v?*v?
 
*
%0
&1
'2
(3
)4
*5
v
%0
&1
'2
(3
)4
*5
+6
,7
-8
.9
/10
011
112
213
314
415
?
regularization_losses
trainable_variables

5layers
6layer_regularization_losses
7metrics
8layer_metrics
9non_trainable_variables
	variables
 
h

%kernel
&bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
h

'kernel
(bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
R
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
h

)kernel
*bias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
 
 
*
%0
&1
'2
(3
)4
*5
*
%0
&1
'2
(3
)4
*5
?
regularization_losses
trainable_variables

Jlayers
Klayer_regularization_losses
Lmetrics
Mlayer_metrics
Nnon_trainable_variables
	variables
 
R
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
h

+kernel
,bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
h

-kernel
.bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
h

/kernel
0bias
[regularization_losses
\trainable_variables
]	variables
^	keras_api
h

1kernel
2bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
R
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
h

3kernel
4bias
gregularization_losses
htrainable_variables
i	variables
j	keras_api
?
kiter

lbeta_1

mbeta_2
	ndecay
olearning_rate+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?
 
 
F
+0
,1
-2
.3
/4
05
16
27
38
49
?
regularization_losses
trainable_variables

players
qlayer_regularization_losses
rmetrics
slayer_metrics
tnon_trainable_variables
	variables
US
VARIABLE_VALUEtraining_334/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEtraining_334/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEtraining_334/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_334/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEtraining_334/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_421/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_421/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_422/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_422/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_423/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_423/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_424/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_424/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_425/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_425/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_426/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_426/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_427/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_427/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_428/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_428/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

u0
 
F
+0
,1
-2
.3
/4
05
16
27
38
49
 

%0
&1

%0
&1
?
:regularization_losses
;trainable_variables
vlayer_regularization_losses

wlayers
xmetrics
ylayer_metrics
znon_trainable_variables
<	variables
 

'0
(1

'0
(1
?
>regularization_losses
?trainable_variables
{layer_regularization_losses

|layers
}metrics
~layer_metrics
non_trainable_variables
@	variables
 
 
 
?
Bregularization_losses
Ctrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
D	variables
 

)0
*1

)0
*1
?
Fregularization_losses
Gtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
H	variables
#
0

1
2
3
4
 

?0
 
 
 
 
 
?
Oregularization_losses
Ptrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
Q	variables
 
 

+0
,1
?
Sregularization_losses
Ttrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
U	variables
 
 

-0
.1
?
Wregularization_losses
Xtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
Y	variables
 
 

/0
01
?
[regularization_losses
\trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
]	variables
 
 

10
21
?
_regularization_losses
`trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
a	variables
 
 
 
?
cregularization_losses
dtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
e	variables
 
 

30
41
?
gregularization_losses
htrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
i	variables
jh
VARIABLE_VALUEtraining_320/Adam/iter>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEtraining_320/Adam/beta_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEtraining_320/Adam/beta_2@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEtraining_320/Adam/decay?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEtraining_320/Adam/learning_rateGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
8
0
1
2
3
4
5
6
7
 

?0
 
F
+0
,1
-2
.3
/4
05
16
27
38
49
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
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
 
 
 
 
 
 
 
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 

+0
,1
 
 
 
 

-0
.1
 
 
 
 

/0
01
 
 
 
 

10
21
 
 
 
 
 
 
 
 
 

30
41
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
SQ
VARIABLE_VALUE	total_3464keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	count_3464keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
hf
VARIABLE_VALUE	total_330Ilayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE	count_330Ilayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
hf
VARIABLE_VALUE	total_331Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE	count_331Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE$training_334/Adam/dense_421/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_334/Adam/dense_421/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_334/Adam/dense_422/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_334/Adam/dense_422/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_334/Adam/dense_423/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_334/Adam/dense_423/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_334/Adam/dense_421/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_334/Adam/dense_421/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_334/Adam/dense_422/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_334/Adam/dense_422/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_334/Adam/dense_423/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_334/Adam/dense_423/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_424/kernel/mWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_424/bias/mWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_425/kernel/mWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_425/bias/mWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_426/kernel/mXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_426/bias/mXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_427/kernel/mXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_427/bias/mXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_428/kernel/mXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_428/bias/mXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_424/kernel/vWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_424/bias/vWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_425/kernel/vWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_425/bias/vWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_426/kernel/vXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_426/bias/vXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_427/kernel/vXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_427/bias/vXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_320/Adam/dense_428/kernel/vXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_320/Adam/dense_428/bias/vXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_13Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13dense_421/kerneldense_421/biasdense_422/kerneldense_422/biasdense_423/kerneldense_423/biasdense_424/kerneldense_424/biasdense_425/kerneldense_425/biasdense_426/kerneldense_426/biasdense_427/kerneldense_427/biasdense_428/kerneldense_428/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_60999786
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*training_334/Adam/iter/Read/ReadVariableOp,training_334/Adam/beta_1/Read/ReadVariableOp,training_334/Adam/beta_2/Read/ReadVariableOp+training_334/Adam/decay/Read/ReadVariableOp3training_334/Adam/learning_rate/Read/ReadVariableOp$dense_421/kernel/Read/ReadVariableOp"dense_421/bias/Read/ReadVariableOp$dense_422/kernel/Read/ReadVariableOp"dense_422/bias/Read/ReadVariableOp$dense_423/kernel/Read/ReadVariableOp"dense_423/bias/Read/ReadVariableOp$dense_424/kernel/Read/ReadVariableOp"dense_424/bias/Read/ReadVariableOp$dense_425/kernel/Read/ReadVariableOp"dense_425/bias/Read/ReadVariableOp$dense_426/kernel/Read/ReadVariableOp"dense_426/bias/Read/ReadVariableOp$dense_427/kernel/Read/ReadVariableOp"dense_427/bias/Read/ReadVariableOp$dense_428/kernel/Read/ReadVariableOp"dense_428/bias/Read/ReadVariableOp*training_320/Adam/iter/Read/ReadVariableOp,training_320/Adam/beta_1/Read/ReadVariableOp,training_320/Adam/beta_2/Read/ReadVariableOp+training_320/Adam/decay/Read/ReadVariableOp3training_320/Adam/learning_rate/Read/ReadVariableOptotal_346/Read/ReadVariableOpcount_346/Read/ReadVariableOptotal_330/Read/ReadVariableOpcount_330/Read/ReadVariableOptotal_331/Read/ReadVariableOpcount_331/Read/ReadVariableOp8training_334/Adam/dense_421/kernel/m/Read/ReadVariableOp6training_334/Adam/dense_421/bias/m/Read/ReadVariableOp8training_334/Adam/dense_422/kernel/m/Read/ReadVariableOp6training_334/Adam/dense_422/bias/m/Read/ReadVariableOp8training_334/Adam/dense_423/kernel/m/Read/ReadVariableOp6training_334/Adam/dense_423/bias/m/Read/ReadVariableOp8training_334/Adam/dense_421/kernel/v/Read/ReadVariableOp6training_334/Adam/dense_421/bias/v/Read/ReadVariableOp8training_334/Adam/dense_422/kernel/v/Read/ReadVariableOp6training_334/Adam/dense_422/bias/v/Read/ReadVariableOp8training_334/Adam/dense_423/kernel/v/Read/ReadVariableOp6training_334/Adam/dense_423/bias/v/Read/ReadVariableOp8training_320/Adam/dense_424/kernel/m/Read/ReadVariableOp6training_320/Adam/dense_424/bias/m/Read/ReadVariableOp8training_320/Adam/dense_425/kernel/m/Read/ReadVariableOp6training_320/Adam/dense_425/bias/m/Read/ReadVariableOp8training_320/Adam/dense_426/kernel/m/Read/ReadVariableOp6training_320/Adam/dense_426/bias/m/Read/ReadVariableOp8training_320/Adam/dense_427/kernel/m/Read/ReadVariableOp6training_320/Adam/dense_427/bias/m/Read/ReadVariableOp8training_320/Adam/dense_428/kernel/m/Read/ReadVariableOp6training_320/Adam/dense_428/bias/m/Read/ReadVariableOp8training_320/Adam/dense_424/kernel/v/Read/ReadVariableOp6training_320/Adam/dense_424/bias/v/Read/ReadVariableOp8training_320/Adam/dense_425/kernel/v/Read/ReadVariableOp6training_320/Adam/dense_425/bias/v/Read/ReadVariableOp8training_320/Adam/dense_426/kernel/v/Read/ReadVariableOp6training_320/Adam/dense_426/bias/v/Read/ReadVariableOp8training_320/Adam/dense_427/kernel/v/Read/ReadVariableOp6training_320/Adam/dense_427/bias/v/Read/ReadVariableOp8training_320/Adam/dense_428/kernel/v/Read/ReadVariableOp6training_320/Adam/dense_428/bias/v/Read/ReadVariableOpConst*M
TinF
D2B		*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_61001454
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametraining_334/Adam/itertraining_334/Adam/beta_1training_334/Adam/beta_2training_334/Adam/decaytraining_334/Adam/learning_ratedense_421/kerneldense_421/biasdense_422/kerneldense_422/biasdense_423/kerneldense_423/biasdense_424/kerneldense_424/biasdense_425/kerneldense_425/biasdense_426/kerneldense_426/biasdense_427/kerneldense_427/biasdense_428/kerneldense_428/biastraining_320/Adam/itertraining_320/Adam/beta_1training_320/Adam/beta_2training_320/Adam/decaytraining_320/Adam/learning_rate	total_346	count_346	total_330	count_330	total_331	count_331$training_334/Adam/dense_421/kernel/m"training_334/Adam/dense_421/bias/m$training_334/Adam/dense_422/kernel/m"training_334/Adam/dense_422/bias/m$training_334/Adam/dense_423/kernel/m"training_334/Adam/dense_423/bias/m$training_334/Adam/dense_421/kernel/v"training_334/Adam/dense_421/bias/v$training_334/Adam/dense_422/kernel/v"training_334/Adam/dense_422/bias/v$training_334/Adam/dense_423/kernel/v"training_334/Adam/dense_423/bias/v$training_320/Adam/dense_424/kernel/m"training_320/Adam/dense_424/bias/m$training_320/Adam/dense_425/kernel/m"training_320/Adam/dense_425/bias/m$training_320/Adam/dense_426/kernel/m"training_320/Adam/dense_426/bias/m$training_320/Adam/dense_427/kernel/m"training_320/Adam/dense_427/bias/m$training_320/Adam/dense_428/kernel/m"training_320/Adam/dense_428/bias/m$training_320/Adam/dense_424/kernel/v"training_320/Adam/dense_424/bias/v$training_320/Adam/dense_425/kernel/v"training_320/Adam/dense_425/bias/v$training_320/Adam/dense_426/kernel/v"training_320/Adam/dense_426/bias/v$training_320/Adam/dense_427/kernel/v"training_320/Adam/dense_427/bias/v$training_320/Adam/dense_428/kernel/v"training_320/Adam/dense_428/bias/v*L
TinE
C2A*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_61001656??
? 
?
G__inference_dense_426_layer_call_and_return_conditional_losses_60999222

inputs-
)tensordot_readvariableop_dense_426_kernel)
%biasadd_readvariableop_dense_426_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_426_kernel* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_426_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?q
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999497

inputs7
3dense_421_tensordot_readvariableop_dense_421_kernel3
/dense_421_biasadd_readvariableop_dense_421_bias7
3dense_422_tensordot_readvariableop_dense_422_kernel3
/dense_422_biasadd_readvariableop_dense_422_bias7
3dense_423_tensordot_readvariableop_dense_423_kernel3
/dense_423_biasadd_readvariableop_dense_423_bias
identity?? dense_421/BiasAdd/ReadVariableOp?"dense_421/Tensordot/ReadVariableOp? dense_422/BiasAdd/ReadVariableOp?"dense_422/Tensordot/ReadVariableOp? dense_423/BiasAdd/ReadVariableOp?"dense_423/Tensordot/ReadVariableOp?
"dense_421/Tensordot/ReadVariableOpReadVariableOp3dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02$
"dense_421/Tensordot/ReadVariableOp~
dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_421/Tensordot/axes?
dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_421/Tensordot/freel
dense_421/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_421/Tensordot/Shape?
!dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/GatherV2/axis?
dense_421/Tensordot/GatherV2GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/free:output:0*dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_421/Tensordot/GatherV2?
#dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_421/Tensordot/GatherV2_1/axis?
dense_421/Tensordot/GatherV2_1GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/axes:output:0,dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_421/Tensordot/GatherV2_1?
dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const?
dense_421/Tensordot/ProdProd%dense_421/Tensordot/GatherV2:output:0"dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod?
dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_1?
dense_421/Tensordot/Prod_1Prod'dense_421/Tensordot/GatherV2_1:output:0$dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod_1?
dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_421/Tensordot/concat/axis?
dense_421/Tensordot/concatConcatV2!dense_421/Tensordot/free:output:0!dense_421/Tensordot/axes:output:0(dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat?
dense_421/Tensordot/stackPack!dense_421/Tensordot/Prod:output:0#dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/stack?
dense_421/Tensordot/transpose	Transposeinputs#dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_421/Tensordot/transpose?
dense_421/Tensordot/ReshapeReshape!dense_421/Tensordot/transpose:y:0"dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_421/Tensordot/Reshape?
dense_421/Tensordot/MatMulMatMul$dense_421/Tensordot/Reshape:output:0*dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_421/Tensordot/MatMul?
dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_2?
!dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/concat_1/axis?
dense_421/Tensordot/concat_1ConcatV2%dense_421/Tensordot/GatherV2:output:0$dense_421/Tensordot/Const_2:output:0*dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat_1?
dense_421/TensordotReshape$dense_421/Tensordot/MatMul:product:0%dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tensordot?
 dense_421/BiasAdd/ReadVariableOpReadVariableOp/dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02"
 dense_421/BiasAdd/ReadVariableOp?
dense_421/BiasAddBiasAdddense_421/Tensordot:output:0(dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_421/BiasAddz
dense_421/TanhTanhdense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tanh?
"dense_422/Tensordot/ReadVariableOpReadVariableOp3dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_422/Tensordot/ReadVariableOp~
dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_422/Tensordot/axes?
dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_422/Tensordot/freex
dense_422/Tensordot/ShapeShapedense_421/Tanh:y:0*
T0*
_output_shapes
:2
dense_422/Tensordot/Shape?
!dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/GatherV2/axis?
dense_422/Tensordot/GatherV2GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/free:output:0*dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_422/Tensordot/GatherV2?
#dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_422/Tensordot/GatherV2_1/axis?
dense_422/Tensordot/GatherV2_1GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/axes:output:0,dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_422/Tensordot/GatherV2_1?
dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const?
dense_422/Tensordot/ProdProd%dense_422/Tensordot/GatherV2:output:0"dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod?
dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const_1?
dense_422/Tensordot/Prod_1Prod'dense_422/Tensordot/GatherV2_1:output:0$dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod_1?
dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_422/Tensordot/concat/axis?
dense_422/Tensordot/concatConcatV2!dense_422/Tensordot/free:output:0!dense_422/Tensordot/axes:output:0(dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat?
dense_422/Tensordot/stackPack!dense_422/Tensordot/Prod:output:0#dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/stack?
dense_422/Tensordot/transpose	Transposedense_421/Tanh:y:0#dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_422/Tensordot/transpose?
dense_422/Tensordot/ReshapeReshape!dense_422/Tensordot/transpose:y:0"dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_422/Tensordot/Reshape?
dense_422/Tensordot/MatMulMatMul$dense_422/Tensordot/Reshape:output:0*dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_422/Tensordot/MatMul?
dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_422/Tensordot/Const_2?
!dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/concat_1/axis?
dense_422/Tensordot/concat_1ConcatV2%dense_422/Tensordot/GatherV2:output:0$dense_422/Tensordot/Const_2:output:0*dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat_1?
dense_422/TensordotReshape$dense_422/Tensordot/MatMul:product:0%dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tensordot?
 dense_422/BiasAdd/ReadVariableOpReadVariableOp/dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02"
 dense_422/BiasAdd/ReadVariableOp?
dense_422/BiasAddBiasAdddense_422/Tensordot:output:0(dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_422/BiasAdd{
dense_422/TanhTanhdense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tanht

add_53/addAddV2inputsdense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2

add_53/add?
"dense_423/Tensordot/ReadVariableOpReadVariableOp3dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02$
"dense_423/Tensordot/ReadVariableOp~
dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_423/Tensordot/axes?
dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_423/Tensordot/freet
dense_423/Tensordot/ShapeShapeadd_53/add:z:0*
T0*
_output_shapes
:2
dense_423/Tensordot/Shape?
!dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/GatherV2/axis?
dense_423/Tensordot/GatherV2GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/free:output:0*dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_423/Tensordot/GatherV2?
#dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_423/Tensordot/GatherV2_1/axis?
dense_423/Tensordot/GatherV2_1GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/axes:output:0,dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_423/Tensordot/GatherV2_1?
dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const?
dense_423/Tensordot/ProdProd%dense_423/Tensordot/GatherV2:output:0"dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod?
dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const_1?
dense_423/Tensordot/Prod_1Prod'dense_423/Tensordot/GatherV2_1:output:0$dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod_1?
dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_423/Tensordot/concat/axis?
dense_423/Tensordot/concatConcatV2!dense_423/Tensordot/free:output:0!dense_423/Tensordot/axes:output:0(dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat?
dense_423/Tensordot/stackPack!dense_423/Tensordot/Prod:output:0#dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/stack?
dense_423/Tensordot/transpose	Transposeadd_53/add:z:0#dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot/transpose?
dense_423/Tensordot/ReshapeReshape!dense_423/Tensordot/transpose:y:0"dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_423/Tensordot/Reshape?
dense_423/Tensordot/MatMulMatMul$dense_423/Tensordot/Reshape:output:0*dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_423/Tensordot/MatMul?
dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_423/Tensordot/Const_2?
!dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/concat_1/axis?
dense_423/Tensordot/concat_1ConcatV2%dense_423/Tensordot/GatherV2:output:0$dense_423/Tensordot/Const_2:output:0*dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat_1?
dense_423/TensordotReshape$dense_423/Tensordot/MatMul:product:0%dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot?
 dense_423/BiasAdd/ReadVariableOpReadVariableOp/dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02"
 dense_423/BiasAdd/ReadVariableOp?
dense_423/BiasAddBiasAdddense_423/Tensordot:output:0(dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_423/BiasAdd?
IdentityIdentitydense_423/BiasAdd:output:0!^dense_421/BiasAdd/ReadVariableOp#^dense_421/Tensordot/ReadVariableOp!^dense_422/BiasAdd/ReadVariableOp#^dense_422/Tensordot/ReadVariableOp!^dense_423/BiasAdd/ReadVariableOp#^dense_423/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_421/BiasAdd/ReadVariableOp dense_421/BiasAdd/ReadVariableOp2H
"dense_421/Tensordot/ReadVariableOp"dense_421/Tensordot/ReadVariableOp2D
 dense_422/BiasAdd/ReadVariableOp dense_422/BiasAdd/ReadVariableOp2H
"dense_422/Tensordot/ReadVariableOp"dense_422/Tensordot/ReadVariableOp2D
 dense_423/BiasAdd/ReadVariableOp dense_423/BiasAdd/ReadVariableOp2H
"dense_423/Tensordot/ReadVariableOp"dense_423/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_61000614
inputs_0
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0dense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609994972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?
?
G__inference_model_975_layer_call_and_return_conditional_losses_60999744

inputs 
autoencoder_dense_421_kernel
autoencoder_dense_421_bias 
autoencoder_dense_422_kernel
autoencoder_dense_422_bias 
autoencoder_dense_423_kernel
autoencoder_dense_423_bias"
discriminator_dense_424_kernel 
discriminator_dense_424_bias"
discriminator_dense_425_kernel 
discriminator_dense_425_bias"
discriminator_dense_426_kernel 
discriminator_dense_426_bias"
discriminator_dense_427_kernel 
discriminator_dense_427_bias"
discriminator_dense_428_kernel 
discriminator_dense_428_bias
identity??#autoencoder/StatefulPartitionedCall?%discriminator/StatefulPartitionedCall?
#autoencoder/StatefulPartitionedCallStatefulPartitionedCallinputsautoencoder_dense_421_kernelautoencoder_dense_421_biasautoencoder_dense_422_kernelautoencoder_dense_422_biasautoencoder_dense_423_kernelautoencoder_dense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609995822%
#autoencoder/StatefulPartitionedCall?
%discriminator/StatefulPartitionedCallStatefulPartitionedCall,autoencoder/StatefulPartitionedCall:output:0discriminator_dense_424_kerneldiscriminator_dense_424_biasdiscriminator_dense_425_kerneldiscriminator_dense_425_biasdiscriminator_dense_426_kerneldiscriminator_dense_426_biasdiscriminator_dense_427_kerneldiscriminator_dense_427_biasdiscriminator_dense_428_kerneldiscriminator_dense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609993962'
%discriminator/StatefulPartitionedCall?
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0$^autoencoder/StatefulPartitionedCall&^discriminator/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::2J
#autoencoder/StatefulPartitionedCall#autoencoder/StatefulPartitionedCall2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_61000422

inputs
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609990372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_425_layer_call_fn_61001134

inputs
dense_425_kernel
dense_425_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_425_kerneldense_425_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_425_layer_call_and_return_conditional_losses_609991792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_422_layer_call_fn_61000982

inputs
dense_422_kernel
dense_422_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_422_kerneldense_422_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_422_layer_call_and_return_conditional_losses_609989362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
? 
?
G__inference_dense_421_layer_call_and_return_conditional_losses_60998893

inputs-
)tensordot_readvariableop_dense_421_kernel)
%biasadd_readvariableop_dense_421_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999020
input_13
dense_421_dense_421_kernel
dense_421_dense_421_bias
dense_422_dense_422_kernel
dense_422_dense_422_bias
dense_423_dense_423_kernel
dense_423_dense_423_bias
identity??!dense_421/StatefulPartitionedCall?!dense_422/StatefulPartitionedCall?!dense_423/StatefulPartitionedCall?
!dense_421/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_421_dense_421_kerneldense_421_dense_421_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_421_layer_call_and_return_conditional_losses_609988932#
!dense_421/StatefulPartitionedCall?
!dense_422/StatefulPartitionedCallStatefulPartitionedCall*dense_421/StatefulPartitionedCall:output:0dense_422_dense_422_kerneldense_422_dense_422_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_422_layer_call_and_return_conditional_losses_609989362#
!dense_422/StatefulPartitionedCall?
add_53/PartitionedCallPartitionedCallinput_13*dense_422/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_53_layer_call_and_return_conditional_losses_609989542
add_53/PartitionedCall?
!dense_423/StatefulPartitionedCallStatefulPartitionedCalladd_53/PartitionedCall:output:0dense_423_dense_423_kerneldense_423_dense_423_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_423_layer_call_and_return_conditional_losses_609989932#
!dense_423/StatefulPartitionedCall?
IdentityIdentity*dense_423/StatefulPartitionedCall:output:0"^dense_421/StatefulPartitionedCall"^dense_422/StatefulPartitionedCall"^dense_423/StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2F
!dense_421/StatefulPartitionedCall!dense_421/StatefulPartitionedCall2F
!dense_422/StatefulPartitionedCall!dense_422/StatefulPartitionedCall2F
!dense_423/StatefulPartitionedCall!dense_423/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
? 
?
G__inference_dense_424_layer_call_and_return_conditional_losses_61001089

inputs-
)tensordot_readvariableop_dense_424_kernel)
%biasadd_readvariableop_dense_424_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_424_kernel* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_424_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_426_layer_call_fn_61001172

inputs
dense_426_kernel
dense_426_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_426_kerneldense_426_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_426_layer_call_and_return_conditional_losses_609992222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_60999087

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_1_layer_call_fn_61001053

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_609990872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?q
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000411

inputs7
3dense_421_tensordot_readvariableop_dense_421_kernel3
/dense_421_biasadd_readvariableop_dense_421_bias7
3dense_422_tensordot_readvariableop_dense_422_kernel3
/dense_422_biasadd_readvariableop_dense_422_bias7
3dense_423_tensordot_readvariableop_dense_423_kernel3
/dense_423_biasadd_readvariableop_dense_423_bias
identity?? dense_421/BiasAdd/ReadVariableOp?"dense_421/Tensordot/ReadVariableOp? dense_422/BiasAdd/ReadVariableOp?"dense_422/Tensordot/ReadVariableOp? dense_423/BiasAdd/ReadVariableOp?"dense_423/Tensordot/ReadVariableOp?
"dense_421/Tensordot/ReadVariableOpReadVariableOp3dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02$
"dense_421/Tensordot/ReadVariableOp~
dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_421/Tensordot/axes?
dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_421/Tensordot/freel
dense_421/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_421/Tensordot/Shape?
!dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/GatherV2/axis?
dense_421/Tensordot/GatherV2GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/free:output:0*dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_421/Tensordot/GatherV2?
#dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_421/Tensordot/GatherV2_1/axis?
dense_421/Tensordot/GatherV2_1GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/axes:output:0,dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_421/Tensordot/GatherV2_1?
dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const?
dense_421/Tensordot/ProdProd%dense_421/Tensordot/GatherV2:output:0"dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod?
dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_1?
dense_421/Tensordot/Prod_1Prod'dense_421/Tensordot/GatherV2_1:output:0$dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod_1?
dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_421/Tensordot/concat/axis?
dense_421/Tensordot/concatConcatV2!dense_421/Tensordot/free:output:0!dense_421/Tensordot/axes:output:0(dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat?
dense_421/Tensordot/stackPack!dense_421/Tensordot/Prod:output:0#dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/stack?
dense_421/Tensordot/transpose	Transposeinputs#dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_421/Tensordot/transpose?
dense_421/Tensordot/ReshapeReshape!dense_421/Tensordot/transpose:y:0"dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_421/Tensordot/Reshape?
dense_421/Tensordot/MatMulMatMul$dense_421/Tensordot/Reshape:output:0*dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_421/Tensordot/MatMul?
dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_2?
!dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/concat_1/axis?
dense_421/Tensordot/concat_1ConcatV2%dense_421/Tensordot/GatherV2:output:0$dense_421/Tensordot/Const_2:output:0*dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat_1?
dense_421/TensordotReshape$dense_421/Tensordot/MatMul:product:0%dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tensordot?
 dense_421/BiasAdd/ReadVariableOpReadVariableOp/dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02"
 dense_421/BiasAdd/ReadVariableOp?
dense_421/BiasAddBiasAdddense_421/Tensordot:output:0(dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_421/BiasAddz
dense_421/TanhTanhdense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tanh?
"dense_422/Tensordot/ReadVariableOpReadVariableOp3dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_422/Tensordot/ReadVariableOp~
dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_422/Tensordot/axes?
dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_422/Tensordot/freex
dense_422/Tensordot/ShapeShapedense_421/Tanh:y:0*
T0*
_output_shapes
:2
dense_422/Tensordot/Shape?
!dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/GatherV2/axis?
dense_422/Tensordot/GatherV2GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/free:output:0*dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_422/Tensordot/GatherV2?
#dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_422/Tensordot/GatherV2_1/axis?
dense_422/Tensordot/GatherV2_1GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/axes:output:0,dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_422/Tensordot/GatherV2_1?
dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const?
dense_422/Tensordot/ProdProd%dense_422/Tensordot/GatherV2:output:0"dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod?
dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const_1?
dense_422/Tensordot/Prod_1Prod'dense_422/Tensordot/GatherV2_1:output:0$dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod_1?
dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_422/Tensordot/concat/axis?
dense_422/Tensordot/concatConcatV2!dense_422/Tensordot/free:output:0!dense_422/Tensordot/axes:output:0(dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat?
dense_422/Tensordot/stackPack!dense_422/Tensordot/Prod:output:0#dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/stack?
dense_422/Tensordot/transpose	Transposedense_421/Tanh:y:0#dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_422/Tensordot/transpose?
dense_422/Tensordot/ReshapeReshape!dense_422/Tensordot/transpose:y:0"dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_422/Tensordot/Reshape?
dense_422/Tensordot/MatMulMatMul$dense_422/Tensordot/Reshape:output:0*dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_422/Tensordot/MatMul?
dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_422/Tensordot/Const_2?
!dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/concat_1/axis?
dense_422/Tensordot/concat_1ConcatV2%dense_422/Tensordot/GatherV2:output:0$dense_422/Tensordot/Const_2:output:0*dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat_1?
dense_422/TensordotReshape$dense_422/Tensordot/MatMul:product:0%dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tensordot?
 dense_422/BiasAdd/ReadVariableOpReadVariableOp/dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02"
 dense_422/BiasAdd/ReadVariableOp?
dense_422/BiasAddBiasAdddense_422/Tensordot:output:0(dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_422/BiasAdd{
dense_422/TanhTanhdense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tanht

add_53/addAddV2inputsdense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2

add_53/add?
"dense_423/Tensordot/ReadVariableOpReadVariableOp3dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02$
"dense_423/Tensordot/ReadVariableOp~
dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_423/Tensordot/axes?
dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_423/Tensordot/freet
dense_423/Tensordot/ShapeShapeadd_53/add:z:0*
T0*
_output_shapes
:2
dense_423/Tensordot/Shape?
!dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/GatherV2/axis?
dense_423/Tensordot/GatherV2GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/free:output:0*dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_423/Tensordot/GatherV2?
#dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_423/Tensordot/GatherV2_1/axis?
dense_423/Tensordot/GatherV2_1GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/axes:output:0,dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_423/Tensordot/GatherV2_1?
dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const?
dense_423/Tensordot/ProdProd%dense_423/Tensordot/GatherV2:output:0"dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod?
dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const_1?
dense_423/Tensordot/Prod_1Prod'dense_423/Tensordot/GatherV2_1:output:0$dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod_1?
dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_423/Tensordot/concat/axis?
dense_423/Tensordot/concatConcatV2!dense_423/Tensordot/free:output:0!dense_423/Tensordot/axes:output:0(dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat?
dense_423/Tensordot/stackPack!dense_423/Tensordot/Prod:output:0#dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/stack?
dense_423/Tensordot/transpose	Transposeadd_53/add:z:0#dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot/transpose?
dense_423/Tensordot/ReshapeReshape!dense_423/Tensordot/transpose:y:0"dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_423/Tensordot/Reshape?
dense_423/Tensordot/MatMulMatMul$dense_423/Tensordot/Reshape:output:0*dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_423/Tensordot/MatMul?
dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_423/Tensordot/Const_2?
!dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/concat_1/axis?
dense_423/Tensordot/concat_1ConcatV2%dense_423/Tensordot/GatherV2:output:0$dense_423/Tensordot/Const_2:output:0*dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat_1?
dense_423/TensordotReshape$dense_423/Tensordot/MatMul:product:0%dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot?
 dense_423/BiasAdd/ReadVariableOpReadVariableOp/dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02"
 dense_423/BiasAdd/ReadVariableOp?
dense_423/BiasAddBiasAdddense_423/Tensordot:output:0(dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_423/BiasAdd?
IdentityIdentitydense_423/BiasAdd:output:0!^dense_421/BiasAdd/ReadVariableOp#^dense_421/Tensordot/ReadVariableOp!^dense_422/BiasAdd/ReadVariableOp#^dense_422/Tensordot/ReadVariableOp!^dense_423/BiasAdd/ReadVariableOp#^dense_423/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_421/BiasAdd/ReadVariableOp dense_421/BiasAdd/ReadVariableOp2H
"dense_421/Tensordot/ReadVariableOp"dense_421/Tensordot/ReadVariableOp2D
 dense_422/BiasAdd/ReadVariableOp dense_422/BiasAdd/ReadVariableOp2H
"dense_422/Tensordot/ReadVariableOp"dense_422/Tensordot/ReadVariableOp2D
 dense_423/BiasAdd/ReadVariableOp dense_423/BiasAdd/ReadVariableOp2H
"dense_423/Tensordot/ReadVariableOp"dense_423/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
G__inference_model_975_layer_call_and_return_conditional_losses_60999996

inputsC
?autoencoder_dense_421_tensordot_readvariableop_dense_421_kernel?
;autoencoder_dense_421_biasadd_readvariableop_dense_421_biasC
?autoencoder_dense_422_tensordot_readvariableop_dense_422_kernel?
;autoencoder_dense_422_biasadd_readvariableop_dense_422_biasC
?autoencoder_dense_423_tensordot_readvariableop_dense_423_kernel?
;autoencoder_dense_423_biasadd_readvariableop_dense_423_biasE
Adiscriminator_dense_424_tensordot_readvariableop_dense_424_kernelA
=discriminator_dense_424_biasadd_readvariableop_dense_424_biasE
Adiscriminator_dense_425_tensordot_readvariableop_dense_425_kernelA
=discriminator_dense_425_biasadd_readvariableop_dense_425_biasE
Adiscriminator_dense_426_tensordot_readvariableop_dense_426_kernelA
=discriminator_dense_426_biasadd_readvariableop_dense_426_biasE
Adiscriminator_dense_427_tensordot_readvariableop_dense_427_kernelA
=discriminator_dense_427_biasadd_readvariableop_dense_427_biasB
>discriminator_dense_428_matmul_readvariableop_dense_428_kernelA
=discriminator_dense_428_biasadd_readvariableop_dense_428_bias
identity??,autoencoder/dense_421/BiasAdd/ReadVariableOp?.autoencoder/dense_421/Tensordot/ReadVariableOp?,autoencoder/dense_422/BiasAdd/ReadVariableOp?.autoencoder/dense_422/Tensordot/ReadVariableOp?,autoencoder/dense_423/BiasAdd/ReadVariableOp?.autoencoder/dense_423/Tensordot/ReadVariableOp?.discriminator/dense_424/BiasAdd/ReadVariableOp?0discriminator/dense_424/Tensordot/ReadVariableOp?.discriminator/dense_425/BiasAdd/ReadVariableOp?0discriminator/dense_425/Tensordot/ReadVariableOp?.discriminator/dense_426/BiasAdd/ReadVariableOp?0discriminator/dense_426/Tensordot/ReadVariableOp?.discriminator/dense_427/BiasAdd/ReadVariableOp?0discriminator/dense_427/Tensordot/ReadVariableOp?.discriminator/dense_428/BiasAdd/ReadVariableOp?-discriminator/dense_428/MatMul/ReadVariableOp?
.autoencoder/dense_421/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype020
.autoencoder/dense_421/Tensordot/ReadVariableOp?
$autoencoder/dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_421/Tensordot/axes?
$autoencoder/dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_421/Tensordot/free?
%autoencoder/dense_421/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2'
%autoencoder/dense_421/Tensordot/Shape?
-autoencoder/dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_421/Tensordot/GatherV2/axis?
(autoencoder/dense_421/Tensordot/GatherV2GatherV2.autoencoder/dense_421/Tensordot/Shape:output:0-autoencoder/dense_421/Tensordot/free:output:06autoencoder/dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_421/Tensordot/GatherV2?
/autoencoder/dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_421/Tensordot/GatherV2_1/axis?
*autoencoder/dense_421/Tensordot/GatherV2_1GatherV2.autoencoder/dense_421/Tensordot/Shape:output:0-autoencoder/dense_421/Tensordot/axes:output:08autoencoder/dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_421/Tensordot/GatherV2_1?
%autoencoder/dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_421/Tensordot/Const?
$autoencoder/dense_421/Tensordot/ProdProd1autoencoder/dense_421/Tensordot/GatherV2:output:0.autoencoder/dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_421/Tensordot/Prod?
'autoencoder/dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_421/Tensordot/Const_1?
&autoencoder/dense_421/Tensordot/Prod_1Prod3autoencoder/dense_421/Tensordot/GatherV2_1:output:00autoencoder/dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_421/Tensordot/Prod_1?
+autoencoder/dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_421/Tensordot/concat/axis?
&autoencoder/dense_421/Tensordot/concatConcatV2-autoencoder/dense_421/Tensordot/free:output:0-autoencoder/dense_421/Tensordot/axes:output:04autoencoder/dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_421/Tensordot/concat?
%autoencoder/dense_421/Tensordot/stackPack-autoencoder/dense_421/Tensordot/Prod:output:0/autoencoder/dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_421/Tensordot/stack?
)autoencoder/dense_421/Tensordot/transpose	Transposeinputs/autoencoder/dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)autoencoder/dense_421/Tensordot/transpose?
'autoencoder/dense_421/Tensordot/ReshapeReshape-autoencoder/dense_421/Tensordot/transpose:y:0.autoencoder/dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_421/Tensordot/Reshape?
&autoencoder/dense_421/Tensordot/MatMulMatMul0autoencoder/dense_421/Tensordot/Reshape:output:06autoencoder/dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&autoencoder/dense_421/Tensordot/MatMul?
'autoencoder/dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_421/Tensordot/Const_2?
-autoencoder/dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_421/Tensordot/concat_1/axis?
(autoencoder/dense_421/Tensordot/concat_1ConcatV21autoencoder/dense_421/Tensordot/GatherV2:output:00autoencoder/dense_421/Tensordot/Const_2:output:06autoencoder/dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_421/Tensordot/concat_1?
autoencoder/dense_421/TensordotReshape0autoencoder/dense_421/Tensordot/MatMul:product:01autoencoder/dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2!
autoencoder/dense_421/Tensordot?
,autoencoder/dense_421/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02.
,autoencoder/dense_421/BiasAdd/ReadVariableOp?
autoencoder/dense_421/BiasAddBiasAdd(autoencoder/dense_421/Tensordot:output:04autoencoder/dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
autoencoder/dense_421/BiasAdd?
autoencoder/dense_421/TanhTanh&autoencoder/dense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
autoencoder/dense_421/Tanh?
.autoencoder/dense_422/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype020
.autoencoder/dense_422/Tensordot/ReadVariableOp?
$autoencoder/dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_422/Tensordot/axes?
$autoencoder/dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_422/Tensordot/free?
%autoencoder/dense_422/Tensordot/ShapeShapeautoencoder/dense_421/Tanh:y:0*
T0*
_output_shapes
:2'
%autoencoder/dense_422/Tensordot/Shape?
-autoencoder/dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_422/Tensordot/GatherV2/axis?
(autoencoder/dense_422/Tensordot/GatherV2GatherV2.autoencoder/dense_422/Tensordot/Shape:output:0-autoencoder/dense_422/Tensordot/free:output:06autoencoder/dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_422/Tensordot/GatherV2?
/autoencoder/dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_422/Tensordot/GatherV2_1/axis?
*autoencoder/dense_422/Tensordot/GatherV2_1GatherV2.autoencoder/dense_422/Tensordot/Shape:output:0-autoencoder/dense_422/Tensordot/axes:output:08autoencoder/dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_422/Tensordot/GatherV2_1?
%autoencoder/dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_422/Tensordot/Const?
$autoencoder/dense_422/Tensordot/ProdProd1autoencoder/dense_422/Tensordot/GatherV2:output:0.autoencoder/dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_422/Tensordot/Prod?
'autoencoder/dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_422/Tensordot/Const_1?
&autoencoder/dense_422/Tensordot/Prod_1Prod3autoencoder/dense_422/Tensordot/GatherV2_1:output:00autoencoder/dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_422/Tensordot/Prod_1?
+autoencoder/dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_422/Tensordot/concat/axis?
&autoencoder/dense_422/Tensordot/concatConcatV2-autoencoder/dense_422/Tensordot/free:output:0-autoencoder/dense_422/Tensordot/axes:output:04autoencoder/dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_422/Tensordot/concat?
%autoencoder/dense_422/Tensordot/stackPack-autoencoder/dense_422/Tensordot/Prod:output:0/autoencoder/dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_422/Tensordot/stack?
)autoencoder/dense_422/Tensordot/transpose	Transposeautoencoder/dense_421/Tanh:y:0/autoencoder/dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2+
)autoencoder/dense_422/Tensordot/transpose?
'autoencoder/dense_422/Tensordot/ReshapeReshape-autoencoder/dense_422/Tensordot/transpose:y:0.autoencoder/dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_422/Tensordot/Reshape?
&autoencoder/dense_422/Tensordot/MatMulMatMul0autoencoder/dense_422/Tensordot/Reshape:output:06autoencoder/dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/dense_422/Tensordot/MatMul?
'autoencoder/dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'autoencoder/dense_422/Tensordot/Const_2?
-autoencoder/dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_422/Tensordot/concat_1/axis?
(autoencoder/dense_422/Tensordot/concat_1ConcatV21autoencoder/dense_422/Tensordot/GatherV2:output:00autoencoder/dense_422/Tensordot/Const_2:output:06autoencoder/dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_422/Tensordot/concat_1?
autoencoder/dense_422/TensordotReshape0autoencoder/dense_422/Tensordot/MatMul:product:01autoencoder/dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
autoencoder/dense_422/Tensordot?
,autoencoder/dense_422/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02.
,autoencoder/dense_422/BiasAdd/ReadVariableOp?
autoencoder/dense_422/BiasAddBiasAdd(autoencoder/dense_422/Tensordot:output:04autoencoder/dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_422/BiasAdd?
autoencoder/dense_422/TanhTanh&autoencoder/dense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_422/Tanh?
autoencoder/add_53/addAddV2inputsautoencoder/dense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2
autoencoder/add_53/add?
.autoencoder/dense_423/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype020
.autoencoder/dense_423/Tensordot/ReadVariableOp?
$autoencoder/dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_423/Tensordot/axes?
$autoencoder/dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_423/Tensordot/free?
%autoencoder/dense_423/Tensordot/ShapeShapeautoencoder/add_53/add:z:0*
T0*
_output_shapes
:2'
%autoencoder/dense_423/Tensordot/Shape?
-autoencoder/dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_423/Tensordot/GatherV2/axis?
(autoencoder/dense_423/Tensordot/GatherV2GatherV2.autoencoder/dense_423/Tensordot/Shape:output:0-autoencoder/dense_423/Tensordot/free:output:06autoencoder/dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_423/Tensordot/GatherV2?
/autoencoder/dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_423/Tensordot/GatherV2_1/axis?
*autoencoder/dense_423/Tensordot/GatherV2_1GatherV2.autoencoder/dense_423/Tensordot/Shape:output:0-autoencoder/dense_423/Tensordot/axes:output:08autoencoder/dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_423/Tensordot/GatherV2_1?
%autoencoder/dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_423/Tensordot/Const?
$autoencoder/dense_423/Tensordot/ProdProd1autoencoder/dense_423/Tensordot/GatherV2:output:0.autoencoder/dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_423/Tensordot/Prod?
'autoencoder/dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_423/Tensordot/Const_1?
&autoencoder/dense_423/Tensordot/Prod_1Prod3autoencoder/dense_423/Tensordot/GatherV2_1:output:00autoencoder/dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_423/Tensordot/Prod_1?
+autoencoder/dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_423/Tensordot/concat/axis?
&autoencoder/dense_423/Tensordot/concatConcatV2-autoencoder/dense_423/Tensordot/free:output:0-autoencoder/dense_423/Tensordot/axes:output:04autoencoder/dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_423/Tensordot/concat?
%autoencoder/dense_423/Tensordot/stackPack-autoencoder/dense_423/Tensordot/Prod:output:0/autoencoder/dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_423/Tensordot/stack?
)autoencoder/dense_423/Tensordot/transpose	Transposeautoencoder/add_53/add:z:0/autoencoder/dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)autoencoder/dense_423/Tensordot/transpose?
'autoencoder/dense_423/Tensordot/ReshapeReshape-autoencoder/dense_423/Tensordot/transpose:y:0.autoencoder/dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_423/Tensordot/Reshape?
&autoencoder/dense_423/Tensordot/MatMulMatMul0autoencoder/dense_423/Tensordot/Reshape:output:06autoencoder/dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/dense_423/Tensordot/MatMul?
'autoencoder/dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'autoencoder/dense_423/Tensordot/Const_2?
-autoencoder/dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_423/Tensordot/concat_1/axis?
(autoencoder/dense_423/Tensordot/concat_1ConcatV21autoencoder/dense_423/Tensordot/GatherV2:output:00autoencoder/dense_423/Tensordot/Const_2:output:06autoencoder/dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_423/Tensordot/concat_1?
autoencoder/dense_423/TensordotReshape0autoencoder/dense_423/Tensordot/MatMul:product:01autoencoder/dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
autoencoder/dense_423/Tensordot?
,autoencoder/dense_423/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02.
,autoencoder/dense_423/BiasAdd/ReadVariableOp?
autoencoder/dense_423/BiasAddBiasAdd(autoencoder/dense_423/Tensordot:output:04autoencoder/dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_423/BiasAdd?
%discriminator/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%discriminator/dropout_1/dropout/Const?
#discriminator/dropout_1/dropout/MulMul&autoencoder/dense_423/BiasAdd:output:0.discriminator/dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2%
#discriminator/dropout_1/dropout/Mul?
%discriminator/dropout_1/dropout/ShapeShape&autoencoder/dense_423/BiasAdd:output:0*
T0*
_output_shapes
:2'
%discriminator/dropout_1/dropout/Shape?
<discriminator/dropout_1/dropout/random_uniform/RandomUniformRandomUniform.discriminator/dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02>
<discriminator/dropout_1/dropout/random_uniform/RandomUniform?
.discriminator/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>20
.discriminator/dropout_1/dropout/GreaterEqual/y?
,discriminator/dropout_1/dropout/GreaterEqualGreaterEqualEdiscriminator/dropout_1/dropout/random_uniform/RandomUniform:output:07discriminator/dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2.
,discriminator/dropout_1/dropout/GreaterEqual?
$discriminator/dropout_1/dropout/CastCast0discriminator/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2&
$discriminator/dropout_1/dropout/Cast?
%discriminator/dropout_1/dropout/Mul_1Mul'discriminator/dropout_1/dropout/Mul:z:0(discriminator/dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2'
%discriminator/dropout_1/dropout/Mul_1?
0discriminator/dense_424/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_424_tensordot_readvariableop_dense_424_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_424/Tensordot/ReadVariableOp?
&discriminator/dense_424/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_424/Tensordot/axes?
&discriminator/dense_424/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_424/Tensordot/free?
'discriminator/dense_424/Tensordot/ShapeShape)discriminator/dropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2)
'discriminator/dense_424/Tensordot/Shape?
/discriminator/dense_424/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_424/Tensordot/GatherV2/axis?
*discriminator/dense_424/Tensordot/GatherV2GatherV20discriminator/dense_424/Tensordot/Shape:output:0/discriminator/dense_424/Tensordot/free:output:08discriminator/dense_424/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_424/Tensordot/GatherV2?
1discriminator/dense_424/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_424/Tensordot/GatherV2_1/axis?
,discriminator/dense_424/Tensordot/GatherV2_1GatherV20discriminator/dense_424/Tensordot/Shape:output:0/discriminator/dense_424/Tensordot/axes:output:0:discriminator/dense_424/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_424/Tensordot/GatherV2_1?
'discriminator/dense_424/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_424/Tensordot/Const?
&discriminator/dense_424/Tensordot/ProdProd3discriminator/dense_424/Tensordot/GatherV2:output:00discriminator/dense_424/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_424/Tensordot/Prod?
)discriminator/dense_424/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_424/Tensordot/Const_1?
(discriminator/dense_424/Tensordot/Prod_1Prod5discriminator/dense_424/Tensordot/GatherV2_1:output:02discriminator/dense_424/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_424/Tensordot/Prod_1?
-discriminator/dense_424/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_424/Tensordot/concat/axis?
(discriminator/dense_424/Tensordot/concatConcatV2/discriminator/dense_424/Tensordot/free:output:0/discriminator/dense_424/Tensordot/axes:output:06discriminator/dense_424/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_424/Tensordot/concat?
'discriminator/dense_424/Tensordot/stackPack/discriminator/dense_424/Tensordot/Prod:output:01discriminator/dense_424/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_424/Tensordot/stack?
+discriminator/dense_424/Tensordot/transpose	Transpose)discriminator/dropout_1/dropout/Mul_1:z:01discriminator/dense_424/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_424/Tensordot/transpose?
)discriminator/dense_424/Tensordot/ReshapeReshape/discriminator/dense_424/Tensordot/transpose:y:00discriminator/dense_424/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_424/Tensordot/Reshape?
(discriminator/dense_424/Tensordot/MatMulMatMul2discriminator/dense_424/Tensordot/Reshape:output:08discriminator/dense_424/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_424/Tensordot/MatMul?
)discriminator/dense_424/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_424/Tensordot/Const_2?
/discriminator/dense_424/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_424/Tensordot/concat_1/axis?
*discriminator/dense_424/Tensordot/concat_1ConcatV23discriminator/dense_424/Tensordot/GatherV2:output:02discriminator/dense_424/Tensordot/Const_2:output:08discriminator/dense_424/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_424/Tensordot/concat_1?
!discriminator/dense_424/TensordotReshape2discriminator/dense_424/Tensordot/MatMul:product:03discriminator/dense_424/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_424/Tensordot?
.discriminator/dense_424/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_424_biasadd_readvariableop_dense_424_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_424/BiasAdd/ReadVariableOp?
discriminator/dense_424/BiasAddBiasAdd*discriminator/dense_424/Tensordot:output:06discriminator/dense_424/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_424/BiasAdd?
discriminator/dense_424/TanhTanh(discriminator/dense_424/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_424/Tanh?
0discriminator/dense_425/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_425_tensordot_readvariableop_dense_425_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_425/Tensordot/ReadVariableOp?
&discriminator/dense_425/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_425/Tensordot/axes?
&discriminator/dense_425/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_425/Tensordot/free?
'discriminator/dense_425/Tensordot/ShapeShape discriminator/dense_424/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_425/Tensordot/Shape?
/discriminator/dense_425/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_425/Tensordot/GatherV2/axis?
*discriminator/dense_425/Tensordot/GatherV2GatherV20discriminator/dense_425/Tensordot/Shape:output:0/discriminator/dense_425/Tensordot/free:output:08discriminator/dense_425/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_425/Tensordot/GatherV2?
1discriminator/dense_425/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_425/Tensordot/GatherV2_1/axis?
,discriminator/dense_425/Tensordot/GatherV2_1GatherV20discriminator/dense_425/Tensordot/Shape:output:0/discriminator/dense_425/Tensordot/axes:output:0:discriminator/dense_425/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_425/Tensordot/GatherV2_1?
'discriminator/dense_425/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_425/Tensordot/Const?
&discriminator/dense_425/Tensordot/ProdProd3discriminator/dense_425/Tensordot/GatherV2:output:00discriminator/dense_425/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_425/Tensordot/Prod?
)discriminator/dense_425/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_425/Tensordot/Const_1?
(discriminator/dense_425/Tensordot/Prod_1Prod5discriminator/dense_425/Tensordot/GatherV2_1:output:02discriminator/dense_425/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_425/Tensordot/Prod_1?
-discriminator/dense_425/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_425/Tensordot/concat/axis?
(discriminator/dense_425/Tensordot/concatConcatV2/discriminator/dense_425/Tensordot/free:output:0/discriminator/dense_425/Tensordot/axes:output:06discriminator/dense_425/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_425/Tensordot/concat?
'discriminator/dense_425/Tensordot/stackPack/discriminator/dense_425/Tensordot/Prod:output:01discriminator/dense_425/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_425/Tensordot/stack?
+discriminator/dense_425/Tensordot/transpose	Transpose discriminator/dense_424/Tanh:y:01discriminator/dense_425/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_425/Tensordot/transpose?
)discriminator/dense_425/Tensordot/ReshapeReshape/discriminator/dense_425/Tensordot/transpose:y:00discriminator/dense_425/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_425/Tensordot/Reshape?
(discriminator/dense_425/Tensordot/MatMulMatMul2discriminator/dense_425/Tensordot/Reshape:output:08discriminator/dense_425/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_425/Tensordot/MatMul?
)discriminator/dense_425/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_425/Tensordot/Const_2?
/discriminator/dense_425/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_425/Tensordot/concat_1/axis?
*discriminator/dense_425/Tensordot/concat_1ConcatV23discriminator/dense_425/Tensordot/GatherV2:output:02discriminator/dense_425/Tensordot/Const_2:output:08discriminator/dense_425/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_425/Tensordot/concat_1?
!discriminator/dense_425/TensordotReshape2discriminator/dense_425/Tensordot/MatMul:product:03discriminator/dense_425/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_425/Tensordot?
.discriminator/dense_425/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_425_biasadd_readvariableop_dense_425_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_425/BiasAdd/ReadVariableOp?
discriminator/dense_425/BiasAddBiasAdd*discriminator/dense_425/Tensordot:output:06discriminator/dense_425/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_425/BiasAdd?
discriminator/dense_425/TanhTanh(discriminator/dense_425/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_425/Tanh?
0discriminator/dense_426/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_426_tensordot_readvariableop_dense_426_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_426/Tensordot/ReadVariableOp?
&discriminator/dense_426/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_426/Tensordot/axes?
&discriminator/dense_426/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_426/Tensordot/free?
'discriminator/dense_426/Tensordot/ShapeShape discriminator/dense_425/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_426/Tensordot/Shape?
/discriminator/dense_426/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_426/Tensordot/GatherV2/axis?
*discriminator/dense_426/Tensordot/GatherV2GatherV20discriminator/dense_426/Tensordot/Shape:output:0/discriminator/dense_426/Tensordot/free:output:08discriminator/dense_426/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_426/Tensordot/GatherV2?
1discriminator/dense_426/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_426/Tensordot/GatherV2_1/axis?
,discriminator/dense_426/Tensordot/GatherV2_1GatherV20discriminator/dense_426/Tensordot/Shape:output:0/discriminator/dense_426/Tensordot/axes:output:0:discriminator/dense_426/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_426/Tensordot/GatherV2_1?
'discriminator/dense_426/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_426/Tensordot/Const?
&discriminator/dense_426/Tensordot/ProdProd3discriminator/dense_426/Tensordot/GatherV2:output:00discriminator/dense_426/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_426/Tensordot/Prod?
)discriminator/dense_426/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_426/Tensordot/Const_1?
(discriminator/dense_426/Tensordot/Prod_1Prod5discriminator/dense_426/Tensordot/GatherV2_1:output:02discriminator/dense_426/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_426/Tensordot/Prod_1?
-discriminator/dense_426/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_426/Tensordot/concat/axis?
(discriminator/dense_426/Tensordot/concatConcatV2/discriminator/dense_426/Tensordot/free:output:0/discriminator/dense_426/Tensordot/axes:output:06discriminator/dense_426/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_426/Tensordot/concat?
'discriminator/dense_426/Tensordot/stackPack/discriminator/dense_426/Tensordot/Prod:output:01discriminator/dense_426/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_426/Tensordot/stack?
+discriminator/dense_426/Tensordot/transpose	Transpose discriminator/dense_425/Tanh:y:01discriminator/dense_426/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_426/Tensordot/transpose?
)discriminator/dense_426/Tensordot/ReshapeReshape/discriminator/dense_426/Tensordot/transpose:y:00discriminator/dense_426/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_426/Tensordot/Reshape?
(discriminator/dense_426/Tensordot/MatMulMatMul2discriminator/dense_426/Tensordot/Reshape:output:08discriminator/dense_426/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_426/Tensordot/MatMul?
)discriminator/dense_426/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_426/Tensordot/Const_2?
/discriminator/dense_426/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_426/Tensordot/concat_1/axis?
*discriminator/dense_426/Tensordot/concat_1ConcatV23discriminator/dense_426/Tensordot/GatherV2:output:02discriminator/dense_426/Tensordot/Const_2:output:08discriminator/dense_426/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_426/Tensordot/concat_1?
!discriminator/dense_426/TensordotReshape2discriminator/dense_426/Tensordot/MatMul:product:03discriminator/dense_426/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_426/Tensordot?
.discriminator/dense_426/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_426_biasadd_readvariableop_dense_426_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_426/BiasAdd/ReadVariableOp?
discriminator/dense_426/BiasAddBiasAdd*discriminator/dense_426/Tensordot:output:06discriminator/dense_426/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_426/BiasAdd?
discriminator/dense_426/TanhTanh(discriminator/dense_426/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_426/Tanh?
0discriminator/dense_427/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_427_tensordot_readvariableop_dense_427_kernel*
_output_shapes
:	?*
dtype022
0discriminator/dense_427/Tensordot/ReadVariableOp?
&discriminator/dense_427/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_427/Tensordot/axes?
&discriminator/dense_427/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_427/Tensordot/free?
'discriminator/dense_427/Tensordot/ShapeShape discriminator/dense_426/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_427/Tensordot/Shape?
/discriminator/dense_427/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_427/Tensordot/GatherV2/axis?
*discriminator/dense_427/Tensordot/GatherV2GatherV20discriminator/dense_427/Tensordot/Shape:output:0/discriminator/dense_427/Tensordot/free:output:08discriminator/dense_427/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_427/Tensordot/GatherV2?
1discriminator/dense_427/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_427/Tensordot/GatherV2_1/axis?
,discriminator/dense_427/Tensordot/GatherV2_1GatherV20discriminator/dense_427/Tensordot/Shape:output:0/discriminator/dense_427/Tensordot/axes:output:0:discriminator/dense_427/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_427/Tensordot/GatherV2_1?
'discriminator/dense_427/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_427/Tensordot/Const?
&discriminator/dense_427/Tensordot/ProdProd3discriminator/dense_427/Tensordot/GatherV2:output:00discriminator/dense_427/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_427/Tensordot/Prod?
)discriminator/dense_427/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_427/Tensordot/Const_1?
(discriminator/dense_427/Tensordot/Prod_1Prod5discriminator/dense_427/Tensordot/GatherV2_1:output:02discriminator/dense_427/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_427/Tensordot/Prod_1?
-discriminator/dense_427/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_427/Tensordot/concat/axis?
(discriminator/dense_427/Tensordot/concatConcatV2/discriminator/dense_427/Tensordot/free:output:0/discriminator/dense_427/Tensordot/axes:output:06discriminator/dense_427/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_427/Tensordot/concat?
'discriminator/dense_427/Tensordot/stackPack/discriminator/dense_427/Tensordot/Prod:output:01discriminator/dense_427/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_427/Tensordot/stack?
+discriminator/dense_427/Tensordot/transpose	Transpose discriminator/dense_426/Tanh:y:01discriminator/dense_427/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_427/Tensordot/transpose?
)discriminator/dense_427/Tensordot/ReshapeReshape/discriminator/dense_427/Tensordot/transpose:y:00discriminator/dense_427/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_427/Tensordot/Reshape?
(discriminator/dense_427/Tensordot/MatMulMatMul2discriminator/dense_427/Tensordot/Reshape:output:08discriminator/dense_427/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(discriminator/dense_427/Tensordot/MatMul?
)discriminator/dense_427/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)discriminator/dense_427/Tensordot/Const_2?
/discriminator/dense_427/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_427/Tensordot/concat_1/axis?
*discriminator/dense_427/Tensordot/concat_1ConcatV23discriminator/dense_427/Tensordot/GatherV2:output:02discriminator/dense_427/Tensordot/Const_2:output:08discriminator/dense_427/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_427/Tensordot/concat_1?
!discriminator/dense_427/TensordotReshape2discriminator/dense_427/Tensordot/MatMul:product:03discriminator/dense_427/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2#
!discriminator/dense_427/Tensordot?
.discriminator/dense_427/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_427_biasadd_readvariableop_dense_427_bias*
_output_shapes
:*
dtype020
.discriminator/dense_427/BiasAdd/ReadVariableOp?
discriminator/dense_427/BiasAddBiasAdd*discriminator/dense_427/Tensordot:output:06discriminator/dense_427/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2!
discriminator/dense_427/BiasAdd?
discriminator/dense_427/TanhTanh(discriminator/dense_427/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
discriminator/dense_427/Tanh?
discriminator/flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2 
discriminator/flatten_52/Const?
 discriminator/flatten_52/ReshapeReshape discriminator/dense_427/Tanh:y:0'discriminator/flatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2"
 discriminator/flatten_52/Reshape?
-discriminator/dense_428/MatMul/ReadVariableOpReadVariableOp>discriminator_dense_428_matmul_readvariableop_dense_428_kernel*
_output_shapes

:*
dtype02/
-discriminator/dense_428/MatMul/ReadVariableOp?
discriminator/dense_428/MatMulMatMul)discriminator/flatten_52/Reshape:output:05discriminator/dense_428/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
discriminator/dense_428/MatMul?
.discriminator/dense_428/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_428_biasadd_readvariableop_dense_428_bias*
_output_shapes
:*
dtype020
.discriminator/dense_428/BiasAdd/ReadVariableOp?
discriminator/dense_428/BiasAddBiasAdd(discriminator/dense_428/MatMul:product:06discriminator/dense_428/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
discriminator/dense_428/BiasAdd?
discriminator/dense_428/SigmoidSigmoid(discriminator/dense_428/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2!
discriminator/dense_428/Sigmoid?
IdentityIdentity#discriminator/dense_428/Sigmoid:y:0-^autoencoder/dense_421/BiasAdd/ReadVariableOp/^autoencoder/dense_421/Tensordot/ReadVariableOp-^autoencoder/dense_422/BiasAdd/ReadVariableOp/^autoencoder/dense_422/Tensordot/ReadVariableOp-^autoencoder/dense_423/BiasAdd/ReadVariableOp/^autoencoder/dense_423/Tensordot/ReadVariableOp/^discriminator/dense_424/BiasAdd/ReadVariableOp1^discriminator/dense_424/Tensordot/ReadVariableOp/^discriminator/dense_425/BiasAdd/ReadVariableOp1^discriminator/dense_425/Tensordot/ReadVariableOp/^discriminator/dense_426/BiasAdd/ReadVariableOp1^discriminator/dense_426/Tensordot/ReadVariableOp/^discriminator/dense_427/BiasAdd/ReadVariableOp1^discriminator/dense_427/Tensordot/ReadVariableOp/^discriminator/dense_428/BiasAdd/ReadVariableOp.^discriminator/dense_428/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::2\
,autoencoder/dense_421/BiasAdd/ReadVariableOp,autoencoder/dense_421/BiasAdd/ReadVariableOp2`
.autoencoder/dense_421/Tensordot/ReadVariableOp.autoencoder/dense_421/Tensordot/ReadVariableOp2\
,autoencoder/dense_422/BiasAdd/ReadVariableOp,autoencoder/dense_422/BiasAdd/ReadVariableOp2`
.autoencoder/dense_422/Tensordot/ReadVariableOp.autoencoder/dense_422/Tensordot/ReadVariableOp2\
,autoencoder/dense_423/BiasAdd/ReadVariableOp,autoencoder/dense_423/BiasAdd/ReadVariableOp2`
.autoencoder/dense_423/Tensordot/ReadVariableOp.autoencoder/dense_423/Tensordot/ReadVariableOp2`
.discriminator/dense_424/BiasAdd/ReadVariableOp.discriminator/dense_424/BiasAdd/ReadVariableOp2d
0discriminator/dense_424/Tensordot/ReadVariableOp0discriminator/dense_424/Tensordot/ReadVariableOp2`
.discriminator/dense_425/BiasAdd/ReadVariableOp.discriminator/dense_425/BiasAdd/ReadVariableOp2d
0discriminator/dense_425/Tensordot/ReadVariableOp0discriminator/dense_425/Tensordot/ReadVariableOp2`
.discriminator/dense_426/BiasAdd/ReadVariableOp.discriminator/dense_426/BiasAdd/ReadVariableOp2d
0discriminator/dense_426/Tensordot/ReadVariableOp0discriminator/dense_426/Tensordot/ReadVariableOp2`
.discriminator/dense_427/BiasAdd/ReadVariableOp.discriminator/dense_427/BiasAdd/ReadVariableOp2d
0discriminator/dense_427/Tensordot/ReadVariableOp0discriminator/dense_427/Tensordot/ReadVariableOp2`
.discriminator/dense_428/BiasAdd/ReadVariableOp.discriminator/dense_428/BiasAdd/ReadVariableOp2^
-discriminator/dense_428/MatMul/ReadVariableOp-discriminator/dense_428/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!__inference__traced_save_61001454
file_prefix5
1savev2_training_334_adam_iter_read_readvariableop	7
3savev2_training_334_adam_beta_1_read_readvariableop7
3savev2_training_334_adam_beta_2_read_readvariableop6
2savev2_training_334_adam_decay_read_readvariableop>
:savev2_training_334_adam_learning_rate_read_readvariableop/
+savev2_dense_421_kernel_read_readvariableop-
)savev2_dense_421_bias_read_readvariableop/
+savev2_dense_422_kernel_read_readvariableop-
)savev2_dense_422_bias_read_readvariableop/
+savev2_dense_423_kernel_read_readvariableop-
)savev2_dense_423_bias_read_readvariableop/
+savev2_dense_424_kernel_read_readvariableop-
)savev2_dense_424_bias_read_readvariableop/
+savev2_dense_425_kernel_read_readvariableop-
)savev2_dense_425_bias_read_readvariableop/
+savev2_dense_426_kernel_read_readvariableop-
)savev2_dense_426_bias_read_readvariableop/
+savev2_dense_427_kernel_read_readvariableop-
)savev2_dense_427_bias_read_readvariableop/
+savev2_dense_428_kernel_read_readvariableop-
)savev2_dense_428_bias_read_readvariableop5
1savev2_training_320_adam_iter_read_readvariableop	7
3savev2_training_320_adam_beta_1_read_readvariableop7
3savev2_training_320_adam_beta_2_read_readvariableop6
2savev2_training_320_adam_decay_read_readvariableop>
:savev2_training_320_adam_learning_rate_read_readvariableop(
$savev2_total_346_read_readvariableop(
$savev2_count_346_read_readvariableop(
$savev2_total_330_read_readvariableop(
$savev2_count_330_read_readvariableop(
$savev2_total_331_read_readvariableop(
$savev2_count_331_read_readvariableopC
?savev2_training_334_adam_dense_421_kernel_m_read_readvariableopA
=savev2_training_334_adam_dense_421_bias_m_read_readvariableopC
?savev2_training_334_adam_dense_422_kernel_m_read_readvariableopA
=savev2_training_334_adam_dense_422_bias_m_read_readvariableopC
?savev2_training_334_adam_dense_423_kernel_m_read_readvariableopA
=savev2_training_334_adam_dense_423_bias_m_read_readvariableopC
?savev2_training_334_adam_dense_421_kernel_v_read_readvariableopA
=savev2_training_334_adam_dense_421_bias_v_read_readvariableopC
?savev2_training_334_adam_dense_422_kernel_v_read_readvariableopA
=savev2_training_334_adam_dense_422_bias_v_read_readvariableopC
?savev2_training_334_adam_dense_423_kernel_v_read_readvariableopA
=savev2_training_334_adam_dense_423_bias_v_read_readvariableopC
?savev2_training_320_adam_dense_424_kernel_m_read_readvariableopA
=savev2_training_320_adam_dense_424_bias_m_read_readvariableopC
?savev2_training_320_adam_dense_425_kernel_m_read_readvariableopA
=savev2_training_320_adam_dense_425_bias_m_read_readvariableopC
?savev2_training_320_adam_dense_426_kernel_m_read_readvariableopA
=savev2_training_320_adam_dense_426_bias_m_read_readvariableopC
?savev2_training_320_adam_dense_427_kernel_m_read_readvariableopA
=savev2_training_320_adam_dense_427_bias_m_read_readvariableopC
?savev2_training_320_adam_dense_428_kernel_m_read_readvariableopA
=savev2_training_320_adam_dense_428_bias_m_read_readvariableopC
?savev2_training_320_adam_dense_424_kernel_v_read_readvariableopA
=savev2_training_320_adam_dense_424_bias_v_read_readvariableopC
?savev2_training_320_adam_dense_425_kernel_v_read_readvariableopA
=savev2_training_320_adam_dense_425_bias_v_read_readvariableopC
?savev2_training_320_adam_dense_426_kernel_v_read_readvariableopA
=savev2_training_320_adam_dense_426_bias_v_read_readvariableopC
?savev2_training_320_adam_dense_427_kernel_v_read_readvariableopA
=savev2_training_320_adam_dense_427_bias_v_read_readvariableopC
?savev2_training_320_adam_dense_428_kernel_v_read_readvariableopA
=savev2_training_320_adam_dense_428_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*?"
value?"B?"AB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*?
value?B?AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_training_334_adam_iter_read_readvariableop3savev2_training_334_adam_beta_1_read_readvariableop3savev2_training_334_adam_beta_2_read_readvariableop2savev2_training_334_adam_decay_read_readvariableop:savev2_training_334_adam_learning_rate_read_readvariableop+savev2_dense_421_kernel_read_readvariableop)savev2_dense_421_bias_read_readvariableop+savev2_dense_422_kernel_read_readvariableop)savev2_dense_422_bias_read_readvariableop+savev2_dense_423_kernel_read_readvariableop)savev2_dense_423_bias_read_readvariableop+savev2_dense_424_kernel_read_readvariableop)savev2_dense_424_bias_read_readvariableop+savev2_dense_425_kernel_read_readvariableop)savev2_dense_425_bias_read_readvariableop+savev2_dense_426_kernel_read_readvariableop)savev2_dense_426_bias_read_readvariableop+savev2_dense_427_kernel_read_readvariableop)savev2_dense_427_bias_read_readvariableop+savev2_dense_428_kernel_read_readvariableop)savev2_dense_428_bias_read_readvariableop1savev2_training_320_adam_iter_read_readvariableop3savev2_training_320_adam_beta_1_read_readvariableop3savev2_training_320_adam_beta_2_read_readvariableop2savev2_training_320_adam_decay_read_readvariableop:savev2_training_320_adam_learning_rate_read_readvariableop$savev2_total_346_read_readvariableop$savev2_count_346_read_readvariableop$savev2_total_330_read_readvariableop$savev2_count_330_read_readvariableop$savev2_total_331_read_readvariableop$savev2_count_331_read_readvariableop?savev2_training_334_adam_dense_421_kernel_m_read_readvariableop=savev2_training_334_adam_dense_421_bias_m_read_readvariableop?savev2_training_334_adam_dense_422_kernel_m_read_readvariableop=savev2_training_334_adam_dense_422_bias_m_read_readvariableop?savev2_training_334_adam_dense_423_kernel_m_read_readvariableop=savev2_training_334_adam_dense_423_bias_m_read_readvariableop?savev2_training_334_adam_dense_421_kernel_v_read_readvariableop=savev2_training_334_adam_dense_421_bias_v_read_readvariableop?savev2_training_334_adam_dense_422_kernel_v_read_readvariableop=savev2_training_334_adam_dense_422_bias_v_read_readvariableop?savev2_training_334_adam_dense_423_kernel_v_read_readvariableop=savev2_training_334_adam_dense_423_bias_v_read_readvariableop?savev2_training_320_adam_dense_424_kernel_m_read_readvariableop=savev2_training_320_adam_dense_424_bias_m_read_readvariableop?savev2_training_320_adam_dense_425_kernel_m_read_readvariableop=savev2_training_320_adam_dense_425_bias_m_read_readvariableop?savev2_training_320_adam_dense_426_kernel_m_read_readvariableop=savev2_training_320_adam_dense_426_bias_m_read_readvariableop?savev2_training_320_adam_dense_427_kernel_m_read_readvariableop=savev2_training_320_adam_dense_427_bias_m_read_readvariableop?savev2_training_320_adam_dense_428_kernel_m_read_readvariableop=savev2_training_320_adam_dense_428_bias_m_read_readvariableop?savev2_training_320_adam_dense_424_kernel_v_read_readvariableop=savev2_training_320_adam_dense_424_bias_v_read_readvariableop?savev2_training_320_adam_dense_425_kernel_v_read_readvariableop=savev2_training_320_adam_dense_425_bias_v_read_readvariableop?savev2_training_320_adam_dense_426_kernel_v_read_readvariableop=savev2_training_320_adam_dense_426_bias_v_read_readvariableop?savev2_training_320_adam_dense_427_kernel_v_read_readvariableop=savev2_training_320_adam_dense_427_bias_v_read_readvariableop?savev2_training_320_adam_dense_428_kernel_v_read_readvariableop=savev2_training_320_adam_dense_428_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	? : :	 ?:?:
??:?:
??:?:
??:?:
??:?:	?:::: : : : : : : : : : : :	? : :	 ?:?:
??:?:	? : :	 ?:?:
??:?:
??:?:
??:?:
??:?:	?::::
??:?:
??:?:
??:?:	?:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	? : 

_output_shapes
: :%!

_output_shapes
:	 ?:!	

_output_shapes	
:?:&
"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :%!!

_output_shapes
:	? : "

_output_shapes
: :%#!

_output_shapes
:	 ?:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	? : (

_output_shapes
: :%)!

_output_shapes
:	 ?:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:%3!

_output_shapes
:	?: 4

_output_shapes
::$5 

_output_shapes

:: 6

_output_shapes
::&7"
 
_output_shapes
:
??:!8

_output_shapes	
:?:&9"
 
_output_shapes
:
??:!:

_output_shapes	
:?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:%=!

_output_shapes
:	?: >

_output_shapes
::$? 

_output_shapes

:: @

_output_shapes
::A

_output_shapes
: 
?%
?
K__inference_discriminator_layer_call_and_return_conditional_losses_60999315
input_14
dense_424_dense_424_kernel
dense_424_dense_424_bias
dense_425_dense_425_kernel
dense_425_dense_425_bias
dense_426_dense_426_kernel
dense_426_dense_426_bias
dense_427_dense_427_kernel
dense_427_dense_427_bias
dense_428_dense_428_kernel
dense_428_dense_428_bias
identity??!dense_424/StatefulPartitionedCall?!dense_425/StatefulPartitionedCall?!dense_426/StatefulPartitionedCall?!dense_427/StatefulPartitionedCall?!dense_428/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallinput_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_609990872#
!dropout_1/StatefulPartitionedCall?
!dense_424/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_424_dense_424_kerneldense_424_dense_424_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_424_layer_call_and_return_conditional_losses_609991362#
!dense_424/StatefulPartitionedCall?
!dense_425/StatefulPartitionedCallStatefulPartitionedCall*dense_424/StatefulPartitionedCall:output:0dense_425_dense_425_kerneldense_425_dense_425_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_425_layer_call_and_return_conditional_losses_609991792#
!dense_425/StatefulPartitionedCall?
!dense_426/StatefulPartitionedCallStatefulPartitionedCall*dense_425/StatefulPartitionedCall:output:0dense_426_dense_426_kerneldense_426_dense_426_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_426_layer_call_and_return_conditional_losses_609992222#
!dense_426/StatefulPartitionedCall?
!dense_427/StatefulPartitionedCallStatefulPartitionedCall*dense_426/StatefulPartitionedCall:output:0dense_427_dense_427_kerneldense_427_dense_427_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_427_layer_call_and_return_conditional_losses_609992652#
!dense_427/StatefulPartitionedCall?
flatten_52/PartitionedCallPartitionedCall*dense_427/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_52_layer_call_and_return_conditional_losses_609992832
flatten_52/PartitionedCall?
!dense_428/StatefulPartitionedCallStatefulPartitionedCall#flatten_52/PartitionedCall:output:0dense_428_dense_428_kerneldense_428_dense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_428_layer_call_and_return_conditional_losses_609993022#
!dense_428/StatefulPartitionedCall?
IdentityIdentity*dense_428/StatefulPartitionedCall:output:0"^dense_424/StatefulPartitionedCall"^dense_425/StatefulPartitionedCall"^dense_426/StatefulPartitionedCall"^dense_427/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::2F
!dense_424/StatefulPartitionedCall!dense_424/StatefulPartitionedCall2F
!dense_425/StatefulPartitionedCall!dense_425/StatefulPartitionedCall2F
!dense_426/StatefulPartitionedCall!dense_426/StatefulPartitionedCall2F
!dense_427/StatefulPartitionedCall!dense_427/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_14
??
?%
$__inference__traced_restore_61001656
file_prefix+
'assignvariableop_training_334_adam_iter/
+assignvariableop_1_training_334_adam_beta_1/
+assignvariableop_2_training_334_adam_beta_2.
*assignvariableop_3_training_334_adam_decay6
2assignvariableop_4_training_334_adam_learning_rate'
#assignvariableop_5_dense_421_kernel%
!assignvariableop_6_dense_421_bias'
#assignvariableop_7_dense_422_kernel%
!assignvariableop_8_dense_422_bias'
#assignvariableop_9_dense_423_kernel&
"assignvariableop_10_dense_423_bias(
$assignvariableop_11_dense_424_kernel&
"assignvariableop_12_dense_424_bias(
$assignvariableop_13_dense_425_kernel&
"assignvariableop_14_dense_425_bias(
$assignvariableop_15_dense_426_kernel&
"assignvariableop_16_dense_426_bias(
$assignvariableop_17_dense_427_kernel&
"assignvariableop_18_dense_427_bias(
$assignvariableop_19_dense_428_kernel&
"assignvariableop_20_dense_428_bias.
*assignvariableop_21_training_320_adam_iter0
,assignvariableop_22_training_320_adam_beta_10
,assignvariableop_23_training_320_adam_beta_2/
+assignvariableop_24_training_320_adam_decay7
3assignvariableop_25_training_320_adam_learning_rate!
assignvariableop_26_total_346!
assignvariableop_27_count_346!
assignvariableop_28_total_330!
assignvariableop_29_count_330!
assignvariableop_30_total_331!
assignvariableop_31_count_331<
8assignvariableop_32_training_334_adam_dense_421_kernel_m:
6assignvariableop_33_training_334_adam_dense_421_bias_m<
8assignvariableop_34_training_334_adam_dense_422_kernel_m:
6assignvariableop_35_training_334_adam_dense_422_bias_m<
8assignvariableop_36_training_334_adam_dense_423_kernel_m:
6assignvariableop_37_training_334_adam_dense_423_bias_m<
8assignvariableop_38_training_334_adam_dense_421_kernel_v:
6assignvariableop_39_training_334_adam_dense_421_bias_v<
8assignvariableop_40_training_334_adam_dense_422_kernel_v:
6assignvariableop_41_training_334_adam_dense_422_bias_v<
8assignvariableop_42_training_334_adam_dense_423_kernel_v:
6assignvariableop_43_training_334_adam_dense_423_bias_v<
8assignvariableop_44_training_320_adam_dense_424_kernel_m:
6assignvariableop_45_training_320_adam_dense_424_bias_m<
8assignvariableop_46_training_320_adam_dense_425_kernel_m:
6assignvariableop_47_training_320_adam_dense_425_bias_m<
8assignvariableop_48_training_320_adam_dense_426_kernel_m:
6assignvariableop_49_training_320_adam_dense_426_bias_m<
8assignvariableop_50_training_320_adam_dense_427_kernel_m:
6assignvariableop_51_training_320_adam_dense_427_bias_m<
8assignvariableop_52_training_320_adam_dense_428_kernel_m:
6assignvariableop_53_training_320_adam_dense_428_bias_m<
8assignvariableop_54_training_320_adam_dense_424_kernel_v:
6assignvariableop_55_training_320_adam_dense_424_bias_v<
8assignvariableop_56_training_320_adam_dense_425_kernel_v:
6assignvariableop_57_training_320_adam_dense_425_bias_v<
8assignvariableop_58_training_320_adam_dense_426_kernel_v:
6assignvariableop_59_training_320_adam_dense_426_bias_v<
8assignvariableop_60_training_320_adam_dense_427_kernel_v:
6assignvariableop_61_training_320_adam_dense_427_bias_v<
8assignvariableop_62_training_320_adam_dense_428_kernel_v:
6assignvariableop_63_training_320_adam_dense_428_bias_v
identity_65??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*?"
value?"B?"AB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*?
value?B?AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_training_334_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_training_334_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp+assignvariableop_2_training_334_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_training_334_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_training_334_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_421_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_421_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_422_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_422_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_423_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_423_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_424_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_424_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dense_425_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_425_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_426_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_426_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_dense_427_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_427_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_dense_428_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_428_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_training_320_adam_iterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_training_320_adam_beta_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_training_320_adam_beta_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_training_320_adam_decayIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp3assignvariableop_25_training_320_adam_learning_rateIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_346Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_346Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_330Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_330Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_331Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_331Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp8assignvariableop_32_training_334_adam_dense_421_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp6assignvariableop_33_training_334_adam_dense_421_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp8assignvariableop_34_training_334_adam_dense_422_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_training_334_adam_dense_422_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp8assignvariableop_36_training_334_adam_dense_423_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp6assignvariableop_37_training_334_adam_dense_423_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp8assignvariableop_38_training_334_adam_dense_421_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_training_334_adam_dense_421_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp8assignvariableop_40_training_334_adam_dense_422_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp6assignvariableop_41_training_334_adam_dense_422_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp8assignvariableop_42_training_334_adam_dense_423_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_training_334_adam_dense_423_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp8assignvariableop_44_training_320_adam_dense_424_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp6assignvariableop_45_training_320_adam_dense_424_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp8assignvariableop_46_training_320_adam_dense_425_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp6assignvariableop_47_training_320_adam_dense_425_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp8assignvariableop_48_training_320_adam_dense_426_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp6assignvariableop_49_training_320_adam_dense_426_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp8assignvariableop_50_training_320_adam_dense_427_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_training_320_adam_dense_427_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp8assignvariableop_52_training_320_adam_dense_428_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp6assignvariableop_53_training_320_adam_dense_428_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp8assignvariableop_54_training_320_adam_dense_424_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp6assignvariableop_55_training_320_adam_dense_424_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp8assignvariableop_56_training_320_adam_dense_425_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp6assignvariableop_57_training_320_adam_dense_425_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp8assignvariableop_58_training_320_adam_dense_426_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp6assignvariableop_59_training_320_adam_dense_426_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp8assignvariableop_60_training_320_adam_dense_427_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp6assignvariableop_61_training_320_adam_dense_427_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp8assignvariableop_62_training_320_adam_dense_428_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp6assignvariableop_63_training_320_adam_dense_428_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_639
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_64?
Identity_65IdentityIdentity_64:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_65"#
identity_65Identity_65:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
G__inference_model_975_layer_call_and_return_conditional_losses_60999676
input_13 
autoencoder_dense_421_kernel
autoencoder_dense_421_bias 
autoencoder_dense_422_kernel
autoencoder_dense_422_bias 
autoencoder_dense_423_kernel
autoencoder_dense_423_bias"
discriminator_dense_424_kernel 
discriminator_dense_424_bias"
discriminator_dense_425_kernel 
discriminator_dense_425_bias"
discriminator_dense_426_kernel 
discriminator_dense_426_bias"
discriminator_dense_427_kernel 
discriminator_dense_427_bias"
discriminator_dense_428_kernel 
discriminator_dense_428_bias
identity??#autoencoder/StatefulPartitionedCall?%discriminator/StatefulPartitionedCall?
#autoencoder/StatefulPartitionedCallStatefulPartitionedCallinput_13autoencoder_dense_421_kernelautoencoder_dense_421_biasautoencoder_dense_422_kernelautoencoder_dense_422_biasautoencoder_dense_423_kernelautoencoder_dense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609995822%
#autoencoder/StatefulPartitionedCall?
%discriminator/StatefulPartitionedCallStatefulPartitionedCall,autoencoder/StatefulPartitionedCall:output:0discriminator_dense_424_kerneldiscriminator_dense_424_biasdiscriminator_dense_425_kerneldiscriminator_dense_425_biasdiscriminator_dense_426_kerneldiscriminator_dense_426_biasdiscriminator_dense_427_kerneldiscriminator_dense_427_biasdiscriminator_dense_428_kerneldiscriminator_dense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609993962'
%discriminator/StatefulPartitionedCall?
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0$^autoencoder/StatefulPartitionedCall&^discriminator/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::2J
#autoencoder/StatefulPartitionedCall#autoencoder/StatefulPartitionedCall2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
?#
?
K__inference_discriminator_layer_call_and_return_conditional_losses_60999396

inputs
dense_424_dense_424_kernel
dense_424_dense_424_bias
dense_425_dense_425_kernel
dense_425_dense_425_bias
dense_426_dense_426_kernel
dense_426_dense_426_bias
dense_427_dense_427_kernel
dense_427_dense_427_bias
dense_428_dense_428_kernel
dense_428_dense_428_bias
identity??!dense_424/StatefulPartitionedCall?!dense_425/StatefulPartitionedCall?!dense_426/StatefulPartitionedCall?!dense_427/StatefulPartitionedCall?!dense_428/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_609990922
dropout_1/PartitionedCall?
!dense_424/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_424_dense_424_kerneldense_424_dense_424_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_424_layer_call_and_return_conditional_losses_609991362#
!dense_424/StatefulPartitionedCall?
!dense_425/StatefulPartitionedCallStatefulPartitionedCall*dense_424/StatefulPartitionedCall:output:0dense_425_dense_425_kerneldense_425_dense_425_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_425_layer_call_and_return_conditional_losses_609991792#
!dense_425/StatefulPartitionedCall?
!dense_426/StatefulPartitionedCallStatefulPartitionedCall*dense_425/StatefulPartitionedCall:output:0dense_426_dense_426_kerneldense_426_dense_426_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_426_layer_call_and_return_conditional_losses_609992222#
!dense_426/StatefulPartitionedCall?
!dense_427/StatefulPartitionedCallStatefulPartitionedCall*dense_426/StatefulPartitionedCall:output:0dense_427_dense_427_kerneldense_427_dense_427_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_427_layer_call_and_return_conditional_losses_609992652#
!dense_427/StatefulPartitionedCall?
flatten_52/PartitionedCallPartitionedCall*dense_427/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_52_layer_call_and_return_conditional_losses_609992832
flatten_52/PartitionedCall?
!dense_428/StatefulPartitionedCallStatefulPartitionedCall#flatten_52/PartitionedCall:output:0dense_428_dense_428_kerneldense_428_dense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_428_layer_call_and_return_conditional_losses_609993022#
!dense_428/StatefulPartitionedCall?
IdentityIdentity*dense_428/StatefulPartitionedCall:output:0"^dense_424/StatefulPartitionedCall"^dense_425/StatefulPartitionedCall"^dense_426/StatefulPartitionedCall"^dense_427/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::2F
!dense_424/StatefulPartitionedCall!dense_424/StatefulPartitionedCall2F
!dense_425/StatefulPartitionedCall!dense_425/StatefulPartitionedCall2F
!dense_426/StatefulPartitionedCall!dense_426/StatefulPartitionedCall2F
!dense_427/StatefulPartitionedCall!dense_427/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_424_layer_call_and_return_conditional_losses_60999136

inputs-
)tensordot_readvariableop_dense_424_kernel)
%biasadd_readvariableop_dense_424_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_424_kernel* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_424_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
0__inference_discriminator_layer_call_fn_60999373
input_14
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14dense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609993602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_14
?
?
,__inference_dense_421_layer_call_fn_61000944

inputs
dense_421_kernel
dense_421_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_421_kerneldense_421_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_421_layer_call_and_return_conditional_losses_609988932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999037

inputs
dense_421_dense_421_kernel
dense_421_dense_421_bias
dense_422_dense_422_kernel
dense_422_dense_422_bias
dense_423_dense_423_kernel
dense_423_dense_423_bias
identity??!dense_421/StatefulPartitionedCall?!dense_422/StatefulPartitionedCall?!dense_423/StatefulPartitionedCall?
!dense_421/StatefulPartitionedCallStatefulPartitionedCallinputsdense_421_dense_421_kerneldense_421_dense_421_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_421_layer_call_and_return_conditional_losses_609988932#
!dense_421/StatefulPartitionedCall?
!dense_422/StatefulPartitionedCallStatefulPartitionedCall*dense_421/StatefulPartitionedCall:output:0dense_422_dense_422_kerneldense_422_dense_422_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_422_layer_call_and_return_conditional_losses_609989362#
!dense_422/StatefulPartitionedCall?
add_53/PartitionedCallPartitionedCallinputs*dense_422/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_53_layer_call_and_return_conditional_losses_609989542
add_53/PartitionedCall?
!dense_423/StatefulPartitionedCallStatefulPartitionedCalladd_53/PartitionedCall:output:0dense_423_dense_423_kerneldense_423_dense_423_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_423_layer_call_and_return_conditional_losses_609989932#
!dense_423/StatefulPartitionedCall?
IdentityIdentity*dense_423/StatefulPartitionedCall:output:0"^dense_421/StatefulPartitionedCall"^dense_422/StatefulPartitionedCall"^dense_423/StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2F
!dense_421/StatefulPartitionedCall!dense_421/StatefulPartitionedCall2F
!dense_422/StatefulPartitionedCall!dense_422/StatefulPartitionedCall2F
!dense_423/StatefulPartitionedCall!dense_423/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?q
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000518
inputs_07
3dense_421_tensordot_readvariableop_dense_421_kernel3
/dense_421_biasadd_readvariableop_dense_421_bias7
3dense_422_tensordot_readvariableop_dense_422_kernel3
/dense_422_biasadd_readvariableop_dense_422_bias7
3dense_423_tensordot_readvariableop_dense_423_kernel3
/dense_423_biasadd_readvariableop_dense_423_bias
identity?? dense_421/BiasAdd/ReadVariableOp?"dense_421/Tensordot/ReadVariableOp? dense_422/BiasAdd/ReadVariableOp?"dense_422/Tensordot/ReadVariableOp? dense_423/BiasAdd/ReadVariableOp?"dense_423/Tensordot/ReadVariableOp?
"dense_421/Tensordot/ReadVariableOpReadVariableOp3dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02$
"dense_421/Tensordot/ReadVariableOp~
dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_421/Tensordot/axes?
dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_421/Tensordot/freen
dense_421/Tensordot/ShapeShapeinputs_0*
T0*
_output_shapes
:2
dense_421/Tensordot/Shape?
!dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/GatherV2/axis?
dense_421/Tensordot/GatherV2GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/free:output:0*dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_421/Tensordot/GatherV2?
#dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_421/Tensordot/GatherV2_1/axis?
dense_421/Tensordot/GatherV2_1GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/axes:output:0,dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_421/Tensordot/GatherV2_1?
dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const?
dense_421/Tensordot/ProdProd%dense_421/Tensordot/GatherV2:output:0"dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod?
dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_1?
dense_421/Tensordot/Prod_1Prod'dense_421/Tensordot/GatherV2_1:output:0$dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod_1?
dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_421/Tensordot/concat/axis?
dense_421/Tensordot/concatConcatV2!dense_421/Tensordot/free:output:0!dense_421/Tensordot/axes:output:0(dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat?
dense_421/Tensordot/stackPack!dense_421/Tensordot/Prod:output:0#dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/stack?
dense_421/Tensordot/transpose	Transposeinputs_0#dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_421/Tensordot/transpose?
dense_421/Tensordot/ReshapeReshape!dense_421/Tensordot/transpose:y:0"dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_421/Tensordot/Reshape?
dense_421/Tensordot/MatMulMatMul$dense_421/Tensordot/Reshape:output:0*dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_421/Tensordot/MatMul?
dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_2?
!dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/concat_1/axis?
dense_421/Tensordot/concat_1ConcatV2%dense_421/Tensordot/GatherV2:output:0$dense_421/Tensordot/Const_2:output:0*dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat_1?
dense_421/TensordotReshape$dense_421/Tensordot/MatMul:product:0%dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tensordot?
 dense_421/BiasAdd/ReadVariableOpReadVariableOp/dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02"
 dense_421/BiasAdd/ReadVariableOp?
dense_421/BiasAddBiasAdddense_421/Tensordot:output:0(dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_421/BiasAddz
dense_421/TanhTanhdense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tanh?
"dense_422/Tensordot/ReadVariableOpReadVariableOp3dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_422/Tensordot/ReadVariableOp~
dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_422/Tensordot/axes?
dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_422/Tensordot/freex
dense_422/Tensordot/ShapeShapedense_421/Tanh:y:0*
T0*
_output_shapes
:2
dense_422/Tensordot/Shape?
!dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/GatherV2/axis?
dense_422/Tensordot/GatherV2GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/free:output:0*dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_422/Tensordot/GatherV2?
#dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_422/Tensordot/GatherV2_1/axis?
dense_422/Tensordot/GatherV2_1GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/axes:output:0,dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_422/Tensordot/GatherV2_1?
dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const?
dense_422/Tensordot/ProdProd%dense_422/Tensordot/GatherV2:output:0"dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod?
dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const_1?
dense_422/Tensordot/Prod_1Prod'dense_422/Tensordot/GatherV2_1:output:0$dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod_1?
dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_422/Tensordot/concat/axis?
dense_422/Tensordot/concatConcatV2!dense_422/Tensordot/free:output:0!dense_422/Tensordot/axes:output:0(dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat?
dense_422/Tensordot/stackPack!dense_422/Tensordot/Prod:output:0#dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/stack?
dense_422/Tensordot/transpose	Transposedense_421/Tanh:y:0#dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_422/Tensordot/transpose?
dense_422/Tensordot/ReshapeReshape!dense_422/Tensordot/transpose:y:0"dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_422/Tensordot/Reshape?
dense_422/Tensordot/MatMulMatMul$dense_422/Tensordot/Reshape:output:0*dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_422/Tensordot/MatMul?
dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_422/Tensordot/Const_2?
!dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/concat_1/axis?
dense_422/Tensordot/concat_1ConcatV2%dense_422/Tensordot/GatherV2:output:0$dense_422/Tensordot/Const_2:output:0*dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat_1?
dense_422/TensordotReshape$dense_422/Tensordot/MatMul:product:0%dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tensordot?
 dense_422/BiasAdd/ReadVariableOpReadVariableOp/dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02"
 dense_422/BiasAdd/ReadVariableOp?
dense_422/BiasAddBiasAdddense_422/Tensordot:output:0(dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_422/BiasAdd{
dense_422/TanhTanhdense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tanhv

add_53/addAddV2inputs_0dense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2

add_53/add?
"dense_423/Tensordot/ReadVariableOpReadVariableOp3dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02$
"dense_423/Tensordot/ReadVariableOp~
dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_423/Tensordot/axes?
dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_423/Tensordot/freet
dense_423/Tensordot/ShapeShapeadd_53/add:z:0*
T0*
_output_shapes
:2
dense_423/Tensordot/Shape?
!dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/GatherV2/axis?
dense_423/Tensordot/GatherV2GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/free:output:0*dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_423/Tensordot/GatherV2?
#dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_423/Tensordot/GatherV2_1/axis?
dense_423/Tensordot/GatherV2_1GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/axes:output:0,dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_423/Tensordot/GatherV2_1?
dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const?
dense_423/Tensordot/ProdProd%dense_423/Tensordot/GatherV2:output:0"dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod?
dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const_1?
dense_423/Tensordot/Prod_1Prod'dense_423/Tensordot/GatherV2_1:output:0$dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod_1?
dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_423/Tensordot/concat/axis?
dense_423/Tensordot/concatConcatV2!dense_423/Tensordot/free:output:0!dense_423/Tensordot/axes:output:0(dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat?
dense_423/Tensordot/stackPack!dense_423/Tensordot/Prod:output:0#dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/stack?
dense_423/Tensordot/transpose	Transposeadd_53/add:z:0#dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot/transpose?
dense_423/Tensordot/ReshapeReshape!dense_423/Tensordot/transpose:y:0"dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_423/Tensordot/Reshape?
dense_423/Tensordot/MatMulMatMul$dense_423/Tensordot/Reshape:output:0*dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_423/Tensordot/MatMul?
dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_423/Tensordot/Const_2?
!dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/concat_1/axis?
dense_423/Tensordot/concat_1ConcatV2%dense_423/Tensordot/GatherV2:output:0$dense_423/Tensordot/Const_2:output:0*dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat_1?
dense_423/TensordotReshape$dense_423/Tensordot/MatMul:product:0%dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot?
 dense_423/BiasAdd/ReadVariableOpReadVariableOp/dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02"
 dense_423/BiasAdd/ReadVariableOp?
dense_423/BiasAddBiasAdddense_423/Tensordot:output:0(dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_423/BiasAdd?
IdentityIdentitydense_423/BiasAdd:output:0!^dense_421/BiasAdd/ReadVariableOp#^dense_421/Tensordot/ReadVariableOp!^dense_422/BiasAdd/ReadVariableOp#^dense_422/Tensordot/ReadVariableOp!^dense_423/BiasAdd/ReadVariableOp#^dense_423/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_421/BiasAdd/ReadVariableOp dense_421/BiasAdd/ReadVariableOp2H
"dense_421/Tensordot/ReadVariableOp"dense_421/Tensordot/ReadVariableOp2D
 dense_422/BiasAdd/ReadVariableOp dense_422/BiasAdd/ReadVariableOp2H
"dense_422/Tensordot/ReadVariableOp"dense_422/Tensordot/ReadVariableOp2D
 dense_423/BiasAdd/ReadVariableOp dense_423/BiasAdd/ReadVariableOp2H
"dense_423/Tensordot/ReadVariableOp"dense_423/Tensordot/ReadVariableOp:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?q
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999582

inputs7
3dense_421_tensordot_readvariableop_dense_421_kernel3
/dense_421_biasadd_readvariableop_dense_421_bias7
3dense_422_tensordot_readvariableop_dense_422_kernel3
/dense_422_biasadd_readvariableop_dense_422_bias7
3dense_423_tensordot_readvariableop_dense_423_kernel3
/dense_423_biasadd_readvariableop_dense_423_bias
identity?? dense_421/BiasAdd/ReadVariableOp?"dense_421/Tensordot/ReadVariableOp? dense_422/BiasAdd/ReadVariableOp?"dense_422/Tensordot/ReadVariableOp? dense_423/BiasAdd/ReadVariableOp?"dense_423/Tensordot/ReadVariableOp?
"dense_421/Tensordot/ReadVariableOpReadVariableOp3dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02$
"dense_421/Tensordot/ReadVariableOp~
dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_421/Tensordot/axes?
dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_421/Tensordot/freel
dense_421/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_421/Tensordot/Shape?
!dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/GatherV2/axis?
dense_421/Tensordot/GatherV2GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/free:output:0*dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_421/Tensordot/GatherV2?
#dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_421/Tensordot/GatherV2_1/axis?
dense_421/Tensordot/GatherV2_1GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/axes:output:0,dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_421/Tensordot/GatherV2_1?
dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const?
dense_421/Tensordot/ProdProd%dense_421/Tensordot/GatherV2:output:0"dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod?
dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_1?
dense_421/Tensordot/Prod_1Prod'dense_421/Tensordot/GatherV2_1:output:0$dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod_1?
dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_421/Tensordot/concat/axis?
dense_421/Tensordot/concatConcatV2!dense_421/Tensordot/free:output:0!dense_421/Tensordot/axes:output:0(dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat?
dense_421/Tensordot/stackPack!dense_421/Tensordot/Prod:output:0#dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/stack?
dense_421/Tensordot/transpose	Transposeinputs#dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_421/Tensordot/transpose?
dense_421/Tensordot/ReshapeReshape!dense_421/Tensordot/transpose:y:0"dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_421/Tensordot/Reshape?
dense_421/Tensordot/MatMulMatMul$dense_421/Tensordot/Reshape:output:0*dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_421/Tensordot/MatMul?
dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_2?
!dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/concat_1/axis?
dense_421/Tensordot/concat_1ConcatV2%dense_421/Tensordot/GatherV2:output:0$dense_421/Tensordot/Const_2:output:0*dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat_1?
dense_421/TensordotReshape$dense_421/Tensordot/MatMul:product:0%dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tensordot?
 dense_421/BiasAdd/ReadVariableOpReadVariableOp/dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02"
 dense_421/BiasAdd/ReadVariableOp?
dense_421/BiasAddBiasAdddense_421/Tensordot:output:0(dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_421/BiasAddz
dense_421/TanhTanhdense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tanh?
"dense_422/Tensordot/ReadVariableOpReadVariableOp3dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_422/Tensordot/ReadVariableOp~
dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_422/Tensordot/axes?
dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_422/Tensordot/freex
dense_422/Tensordot/ShapeShapedense_421/Tanh:y:0*
T0*
_output_shapes
:2
dense_422/Tensordot/Shape?
!dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/GatherV2/axis?
dense_422/Tensordot/GatherV2GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/free:output:0*dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_422/Tensordot/GatherV2?
#dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_422/Tensordot/GatherV2_1/axis?
dense_422/Tensordot/GatherV2_1GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/axes:output:0,dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_422/Tensordot/GatherV2_1?
dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const?
dense_422/Tensordot/ProdProd%dense_422/Tensordot/GatherV2:output:0"dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod?
dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const_1?
dense_422/Tensordot/Prod_1Prod'dense_422/Tensordot/GatherV2_1:output:0$dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod_1?
dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_422/Tensordot/concat/axis?
dense_422/Tensordot/concatConcatV2!dense_422/Tensordot/free:output:0!dense_422/Tensordot/axes:output:0(dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat?
dense_422/Tensordot/stackPack!dense_422/Tensordot/Prod:output:0#dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/stack?
dense_422/Tensordot/transpose	Transposedense_421/Tanh:y:0#dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_422/Tensordot/transpose?
dense_422/Tensordot/ReshapeReshape!dense_422/Tensordot/transpose:y:0"dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_422/Tensordot/Reshape?
dense_422/Tensordot/MatMulMatMul$dense_422/Tensordot/Reshape:output:0*dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_422/Tensordot/MatMul?
dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_422/Tensordot/Const_2?
!dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/concat_1/axis?
dense_422/Tensordot/concat_1ConcatV2%dense_422/Tensordot/GatherV2:output:0$dense_422/Tensordot/Const_2:output:0*dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat_1?
dense_422/TensordotReshape$dense_422/Tensordot/MatMul:product:0%dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tensordot?
 dense_422/BiasAdd/ReadVariableOpReadVariableOp/dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02"
 dense_422/BiasAdd/ReadVariableOp?
dense_422/BiasAddBiasAdddense_422/Tensordot:output:0(dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_422/BiasAdd{
dense_422/TanhTanhdense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tanht

add_53/addAddV2inputsdense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2

add_53/add?
"dense_423/Tensordot/ReadVariableOpReadVariableOp3dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02$
"dense_423/Tensordot/ReadVariableOp~
dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_423/Tensordot/axes?
dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_423/Tensordot/freet
dense_423/Tensordot/ShapeShapeadd_53/add:z:0*
T0*
_output_shapes
:2
dense_423/Tensordot/Shape?
!dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/GatherV2/axis?
dense_423/Tensordot/GatherV2GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/free:output:0*dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_423/Tensordot/GatherV2?
#dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_423/Tensordot/GatherV2_1/axis?
dense_423/Tensordot/GatherV2_1GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/axes:output:0,dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_423/Tensordot/GatherV2_1?
dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const?
dense_423/Tensordot/ProdProd%dense_423/Tensordot/GatherV2:output:0"dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod?
dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const_1?
dense_423/Tensordot/Prod_1Prod'dense_423/Tensordot/GatherV2_1:output:0$dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod_1?
dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_423/Tensordot/concat/axis?
dense_423/Tensordot/concatConcatV2!dense_423/Tensordot/free:output:0!dense_423/Tensordot/axes:output:0(dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat?
dense_423/Tensordot/stackPack!dense_423/Tensordot/Prod:output:0#dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/stack?
dense_423/Tensordot/transpose	Transposeadd_53/add:z:0#dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot/transpose?
dense_423/Tensordot/ReshapeReshape!dense_423/Tensordot/transpose:y:0"dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_423/Tensordot/Reshape?
dense_423/Tensordot/MatMulMatMul$dense_423/Tensordot/Reshape:output:0*dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_423/Tensordot/MatMul?
dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_423/Tensordot/Const_2?
!dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/concat_1/axis?
dense_423/Tensordot/concat_1ConcatV2%dense_423/Tensordot/GatherV2:output:0$dense_423/Tensordot/Const_2:output:0*dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat_1?
dense_423/TensordotReshape$dense_423/Tensordot/MatMul:product:0%dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot?
 dense_423/BiasAdd/ReadVariableOpReadVariableOp/dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02"
 dense_423/BiasAdd/ReadVariableOp?
dense_423/BiasAddBiasAdddense_423/Tensordot:output:0(dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_423/BiasAdd?
IdentityIdentitydense_423/BiasAdd:output:0!^dense_421/BiasAdd/ReadVariableOp#^dense_421/Tensordot/ReadVariableOp!^dense_422/BiasAdd/ReadVariableOp#^dense_422/Tensordot/ReadVariableOp!^dense_423/BiasAdd/ReadVariableOp#^dense_423/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_421/BiasAdd/ReadVariableOp dense_421/BiasAdd/ReadVariableOp2H
"dense_421/Tensordot/ReadVariableOp"dense_421/Tensordot/ReadVariableOp2D
 dense_422/BiasAdd/ReadVariableOp dense_422/BiasAdd/ReadVariableOp2H
"dense_422/Tensordot/ReadVariableOp"dense_422/Tensordot/ReadVariableOp2D
 dense_423/BiasAdd/ReadVariableOp dense_423/BiasAdd/ReadVariableOp2H
"dense_423/Tensordot/ReadVariableOp"dense_423/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_424_layer_call_fn_61001096

inputs
dense_424_kernel
dense_424_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_424_kerneldense_424_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_424_layer_call_and_return_conditional_losses_609991362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_model_975_layer_call_fn_60999720
input_13
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13dense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_biasdense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_975_layer_call_and_return_conditional_losses_609997012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
?
?
.__inference_autoencoder_layer_call_fn_61000433

inputs
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609990622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_427_layer_call_fn_61001210

inputs
dense_427_kernel
dense_427_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_427_kerneldense_427_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_427_layer_call_and_return_conditional_losses_609992652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
#__inference__wrapped_model_60998858
input_13M
Imodel_975_autoencoder_dense_421_tensordot_readvariableop_dense_421_kernelI
Emodel_975_autoencoder_dense_421_biasadd_readvariableop_dense_421_biasM
Imodel_975_autoencoder_dense_422_tensordot_readvariableop_dense_422_kernelI
Emodel_975_autoencoder_dense_422_biasadd_readvariableop_dense_422_biasM
Imodel_975_autoencoder_dense_423_tensordot_readvariableop_dense_423_kernelI
Emodel_975_autoencoder_dense_423_biasadd_readvariableop_dense_423_biasO
Kmodel_975_discriminator_dense_424_tensordot_readvariableop_dense_424_kernelK
Gmodel_975_discriminator_dense_424_biasadd_readvariableop_dense_424_biasO
Kmodel_975_discriminator_dense_425_tensordot_readvariableop_dense_425_kernelK
Gmodel_975_discriminator_dense_425_biasadd_readvariableop_dense_425_biasO
Kmodel_975_discriminator_dense_426_tensordot_readvariableop_dense_426_kernelK
Gmodel_975_discriminator_dense_426_biasadd_readvariableop_dense_426_biasO
Kmodel_975_discriminator_dense_427_tensordot_readvariableop_dense_427_kernelK
Gmodel_975_discriminator_dense_427_biasadd_readvariableop_dense_427_biasL
Hmodel_975_discriminator_dense_428_matmul_readvariableop_dense_428_kernelK
Gmodel_975_discriminator_dense_428_biasadd_readvariableop_dense_428_bias
identity??6model_975/autoencoder/dense_421/BiasAdd/ReadVariableOp?8model_975/autoencoder/dense_421/Tensordot/ReadVariableOp?6model_975/autoencoder/dense_422/BiasAdd/ReadVariableOp?8model_975/autoencoder/dense_422/Tensordot/ReadVariableOp?6model_975/autoencoder/dense_423/BiasAdd/ReadVariableOp?8model_975/autoencoder/dense_423/Tensordot/ReadVariableOp?8model_975/discriminator/dense_424/BiasAdd/ReadVariableOp?:model_975/discriminator/dense_424/Tensordot/ReadVariableOp?8model_975/discriminator/dense_425/BiasAdd/ReadVariableOp?:model_975/discriminator/dense_425/Tensordot/ReadVariableOp?8model_975/discriminator/dense_426/BiasAdd/ReadVariableOp?:model_975/discriminator/dense_426/Tensordot/ReadVariableOp?8model_975/discriminator/dense_427/BiasAdd/ReadVariableOp?:model_975/discriminator/dense_427/Tensordot/ReadVariableOp?8model_975/discriminator/dense_428/BiasAdd/ReadVariableOp?7model_975/discriminator/dense_428/MatMul/ReadVariableOp?
8model_975/autoencoder/dense_421/Tensordot/ReadVariableOpReadVariableOpImodel_975_autoencoder_dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02:
8model_975/autoencoder/dense_421/Tensordot/ReadVariableOp?
.model_975/autoencoder/dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.model_975/autoencoder/dense_421/Tensordot/axes?
.model_975/autoencoder/dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.model_975/autoencoder/dense_421/Tensordot/free?
/model_975/autoencoder/dense_421/Tensordot/ShapeShapeinput_13*
T0*
_output_shapes
:21
/model_975/autoencoder/dense_421/Tensordot/Shape?
7model_975/autoencoder/dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/autoencoder/dense_421/Tensordot/GatherV2/axis?
2model_975/autoencoder/dense_421/Tensordot/GatherV2GatherV28model_975/autoencoder/dense_421/Tensordot/Shape:output:07model_975/autoencoder/dense_421/Tensordot/free:output:0@model_975/autoencoder/dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2model_975/autoencoder/dense_421/Tensordot/GatherV2?
9model_975/autoencoder/dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/autoencoder/dense_421/Tensordot/GatherV2_1/axis?
4model_975/autoencoder/dense_421/Tensordot/GatherV2_1GatherV28model_975/autoencoder/dense_421/Tensordot/Shape:output:07model_975/autoencoder/dense_421/Tensordot/axes:output:0Bmodel_975/autoencoder/dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_975/autoencoder/dense_421/Tensordot/GatherV2_1?
/model_975/autoencoder/dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_975/autoencoder/dense_421/Tensordot/Const?
.model_975/autoencoder/dense_421/Tensordot/ProdProd;model_975/autoencoder/dense_421/Tensordot/GatherV2:output:08model_975/autoencoder/dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.model_975/autoencoder/dense_421/Tensordot/Prod?
1model_975/autoencoder/dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_975/autoencoder/dense_421/Tensordot/Const_1?
0model_975/autoencoder/dense_421/Tensordot/Prod_1Prod=model_975/autoencoder/dense_421/Tensordot/GatherV2_1:output:0:model_975/autoencoder/dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0model_975/autoencoder/dense_421/Tensordot/Prod_1?
5model_975/autoencoder/dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5model_975/autoencoder/dense_421/Tensordot/concat/axis?
0model_975/autoencoder/dense_421/Tensordot/concatConcatV27model_975/autoencoder/dense_421/Tensordot/free:output:07model_975/autoencoder/dense_421/Tensordot/axes:output:0>model_975/autoencoder/dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0model_975/autoencoder/dense_421/Tensordot/concat?
/model_975/autoencoder/dense_421/Tensordot/stackPack7model_975/autoencoder/dense_421/Tensordot/Prod:output:09model_975/autoencoder/dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/model_975/autoencoder/dense_421/Tensordot/stack?
3model_975/autoencoder/dense_421/Tensordot/transpose	Transposeinput_139model_975/autoencoder/dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????25
3model_975/autoencoder/dense_421/Tensordot/transpose?
1model_975/autoencoder/dense_421/Tensordot/ReshapeReshape7model_975/autoencoder/dense_421/Tensordot/transpose:y:08model_975/autoencoder/dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1model_975/autoencoder/dense_421/Tensordot/Reshape?
0model_975/autoencoder/dense_421/Tensordot/MatMulMatMul:model_975/autoencoder/dense_421/Tensordot/Reshape:output:0@model_975/autoencoder/dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 22
0model_975/autoencoder/dense_421/Tensordot/MatMul?
1model_975/autoencoder/dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_975/autoencoder/dense_421/Tensordot/Const_2?
7model_975/autoencoder/dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/autoencoder/dense_421/Tensordot/concat_1/axis?
2model_975/autoencoder/dense_421/Tensordot/concat_1ConcatV2;model_975/autoencoder/dense_421/Tensordot/GatherV2:output:0:model_975/autoencoder/dense_421/Tensordot/Const_2:output:0@model_975/autoencoder/dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2model_975/autoencoder/dense_421/Tensordot/concat_1?
)model_975/autoencoder/dense_421/TensordotReshape:model_975/autoencoder/dense_421/Tensordot/MatMul:product:0;model_975/autoencoder/dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2+
)model_975/autoencoder/dense_421/Tensordot?
6model_975/autoencoder/dense_421/BiasAdd/ReadVariableOpReadVariableOpEmodel_975_autoencoder_dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype028
6model_975/autoencoder/dense_421/BiasAdd/ReadVariableOp?
'model_975/autoencoder/dense_421/BiasAddBiasAdd2model_975/autoencoder/dense_421/Tensordot:output:0>model_975/autoencoder/dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2)
'model_975/autoencoder/dense_421/BiasAdd?
$model_975/autoencoder/dense_421/TanhTanh0model_975/autoencoder/dense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2&
$model_975/autoencoder/dense_421/Tanh?
8model_975/autoencoder/dense_422/Tensordot/ReadVariableOpReadVariableOpImodel_975_autoencoder_dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02:
8model_975/autoencoder/dense_422/Tensordot/ReadVariableOp?
.model_975/autoencoder/dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.model_975/autoencoder/dense_422/Tensordot/axes?
.model_975/autoencoder/dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.model_975/autoencoder/dense_422/Tensordot/free?
/model_975/autoencoder/dense_422/Tensordot/ShapeShape(model_975/autoencoder/dense_421/Tanh:y:0*
T0*
_output_shapes
:21
/model_975/autoencoder/dense_422/Tensordot/Shape?
7model_975/autoencoder/dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/autoencoder/dense_422/Tensordot/GatherV2/axis?
2model_975/autoencoder/dense_422/Tensordot/GatherV2GatherV28model_975/autoencoder/dense_422/Tensordot/Shape:output:07model_975/autoencoder/dense_422/Tensordot/free:output:0@model_975/autoencoder/dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2model_975/autoencoder/dense_422/Tensordot/GatherV2?
9model_975/autoencoder/dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/autoencoder/dense_422/Tensordot/GatherV2_1/axis?
4model_975/autoencoder/dense_422/Tensordot/GatherV2_1GatherV28model_975/autoencoder/dense_422/Tensordot/Shape:output:07model_975/autoencoder/dense_422/Tensordot/axes:output:0Bmodel_975/autoencoder/dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_975/autoencoder/dense_422/Tensordot/GatherV2_1?
/model_975/autoencoder/dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_975/autoencoder/dense_422/Tensordot/Const?
.model_975/autoencoder/dense_422/Tensordot/ProdProd;model_975/autoencoder/dense_422/Tensordot/GatherV2:output:08model_975/autoencoder/dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.model_975/autoencoder/dense_422/Tensordot/Prod?
1model_975/autoencoder/dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_975/autoencoder/dense_422/Tensordot/Const_1?
0model_975/autoencoder/dense_422/Tensordot/Prod_1Prod=model_975/autoencoder/dense_422/Tensordot/GatherV2_1:output:0:model_975/autoencoder/dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0model_975/autoencoder/dense_422/Tensordot/Prod_1?
5model_975/autoencoder/dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5model_975/autoencoder/dense_422/Tensordot/concat/axis?
0model_975/autoencoder/dense_422/Tensordot/concatConcatV27model_975/autoencoder/dense_422/Tensordot/free:output:07model_975/autoencoder/dense_422/Tensordot/axes:output:0>model_975/autoencoder/dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0model_975/autoencoder/dense_422/Tensordot/concat?
/model_975/autoencoder/dense_422/Tensordot/stackPack7model_975/autoencoder/dense_422/Tensordot/Prod:output:09model_975/autoencoder/dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/model_975/autoencoder/dense_422/Tensordot/stack?
3model_975/autoencoder/dense_422/Tensordot/transpose	Transpose(model_975/autoencoder/dense_421/Tanh:y:09model_975/autoencoder/dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 25
3model_975/autoencoder/dense_422/Tensordot/transpose?
1model_975/autoencoder/dense_422/Tensordot/ReshapeReshape7model_975/autoencoder/dense_422/Tensordot/transpose:y:08model_975/autoencoder/dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1model_975/autoencoder/dense_422/Tensordot/Reshape?
0model_975/autoencoder/dense_422/Tensordot/MatMulMatMul:model_975/autoencoder/dense_422/Tensordot/Reshape:output:0@model_975/autoencoder/dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0model_975/autoencoder/dense_422/Tensordot/MatMul?
1model_975/autoencoder/dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?23
1model_975/autoencoder/dense_422/Tensordot/Const_2?
7model_975/autoencoder/dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/autoencoder/dense_422/Tensordot/concat_1/axis?
2model_975/autoencoder/dense_422/Tensordot/concat_1ConcatV2;model_975/autoencoder/dense_422/Tensordot/GatherV2:output:0:model_975/autoencoder/dense_422/Tensordot/Const_2:output:0@model_975/autoencoder/dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2model_975/autoencoder/dense_422/Tensordot/concat_1?
)model_975/autoencoder/dense_422/TensordotReshape:model_975/autoencoder/dense_422/Tensordot/MatMul:product:0;model_975/autoencoder/dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2+
)model_975/autoencoder/dense_422/Tensordot?
6model_975/autoencoder/dense_422/BiasAdd/ReadVariableOpReadVariableOpEmodel_975_autoencoder_dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype028
6model_975/autoencoder/dense_422/BiasAdd/ReadVariableOp?
'model_975/autoencoder/dense_422/BiasAddBiasAdd2model_975/autoencoder/dense_422/Tensordot:output:0>model_975/autoencoder/dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2)
'model_975/autoencoder/dense_422/BiasAdd?
$model_975/autoencoder/dense_422/TanhTanh0model_975/autoencoder/dense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2&
$model_975/autoencoder/dense_422/Tanh?
 model_975/autoencoder/add_53/addAddV2input_13(model_975/autoencoder/dense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2"
 model_975/autoencoder/add_53/add?
8model_975/autoencoder/dense_423/Tensordot/ReadVariableOpReadVariableOpImodel_975_autoencoder_dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02:
8model_975/autoencoder/dense_423/Tensordot/ReadVariableOp?
.model_975/autoencoder/dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.model_975/autoencoder/dense_423/Tensordot/axes?
.model_975/autoencoder/dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.model_975/autoencoder/dense_423/Tensordot/free?
/model_975/autoencoder/dense_423/Tensordot/ShapeShape$model_975/autoencoder/add_53/add:z:0*
T0*
_output_shapes
:21
/model_975/autoencoder/dense_423/Tensordot/Shape?
7model_975/autoencoder/dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/autoencoder/dense_423/Tensordot/GatherV2/axis?
2model_975/autoencoder/dense_423/Tensordot/GatherV2GatherV28model_975/autoencoder/dense_423/Tensordot/Shape:output:07model_975/autoencoder/dense_423/Tensordot/free:output:0@model_975/autoencoder/dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2model_975/autoencoder/dense_423/Tensordot/GatherV2?
9model_975/autoencoder/dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/autoencoder/dense_423/Tensordot/GatherV2_1/axis?
4model_975/autoencoder/dense_423/Tensordot/GatherV2_1GatherV28model_975/autoencoder/dense_423/Tensordot/Shape:output:07model_975/autoencoder/dense_423/Tensordot/axes:output:0Bmodel_975/autoencoder/dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_975/autoencoder/dense_423/Tensordot/GatherV2_1?
/model_975/autoencoder/dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_975/autoencoder/dense_423/Tensordot/Const?
.model_975/autoencoder/dense_423/Tensordot/ProdProd;model_975/autoencoder/dense_423/Tensordot/GatherV2:output:08model_975/autoencoder/dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.model_975/autoencoder/dense_423/Tensordot/Prod?
1model_975/autoencoder/dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_975/autoencoder/dense_423/Tensordot/Const_1?
0model_975/autoencoder/dense_423/Tensordot/Prod_1Prod=model_975/autoencoder/dense_423/Tensordot/GatherV2_1:output:0:model_975/autoencoder/dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0model_975/autoencoder/dense_423/Tensordot/Prod_1?
5model_975/autoencoder/dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5model_975/autoencoder/dense_423/Tensordot/concat/axis?
0model_975/autoencoder/dense_423/Tensordot/concatConcatV27model_975/autoencoder/dense_423/Tensordot/free:output:07model_975/autoencoder/dense_423/Tensordot/axes:output:0>model_975/autoencoder/dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0model_975/autoencoder/dense_423/Tensordot/concat?
/model_975/autoencoder/dense_423/Tensordot/stackPack7model_975/autoencoder/dense_423/Tensordot/Prod:output:09model_975/autoencoder/dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/model_975/autoencoder/dense_423/Tensordot/stack?
3model_975/autoencoder/dense_423/Tensordot/transpose	Transpose$model_975/autoencoder/add_53/add:z:09model_975/autoencoder/dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????25
3model_975/autoencoder/dense_423/Tensordot/transpose?
1model_975/autoencoder/dense_423/Tensordot/ReshapeReshape7model_975/autoencoder/dense_423/Tensordot/transpose:y:08model_975/autoencoder/dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1model_975/autoencoder/dense_423/Tensordot/Reshape?
0model_975/autoencoder/dense_423/Tensordot/MatMulMatMul:model_975/autoencoder/dense_423/Tensordot/Reshape:output:0@model_975/autoencoder/dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0model_975/autoencoder/dense_423/Tensordot/MatMul?
1model_975/autoencoder/dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?23
1model_975/autoencoder/dense_423/Tensordot/Const_2?
7model_975/autoencoder/dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/autoencoder/dense_423/Tensordot/concat_1/axis?
2model_975/autoencoder/dense_423/Tensordot/concat_1ConcatV2;model_975/autoencoder/dense_423/Tensordot/GatherV2:output:0:model_975/autoencoder/dense_423/Tensordot/Const_2:output:0@model_975/autoencoder/dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2model_975/autoencoder/dense_423/Tensordot/concat_1?
)model_975/autoencoder/dense_423/TensordotReshape:model_975/autoencoder/dense_423/Tensordot/MatMul:product:0;model_975/autoencoder/dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2+
)model_975/autoencoder/dense_423/Tensordot?
6model_975/autoencoder/dense_423/BiasAdd/ReadVariableOpReadVariableOpEmodel_975_autoencoder_dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype028
6model_975/autoencoder/dense_423/BiasAdd/ReadVariableOp?
'model_975/autoencoder/dense_423/BiasAddBiasAdd2model_975/autoencoder/dense_423/Tensordot:output:0>model_975/autoencoder/dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2)
'model_975/autoencoder/dense_423/BiasAdd?
*model_975/discriminator/dropout_1/IdentityIdentity0model_975/autoencoder/dense_423/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2,
*model_975/discriminator/dropout_1/Identity?
:model_975/discriminator/dense_424/Tensordot/ReadVariableOpReadVariableOpKmodel_975_discriminator_dense_424_tensordot_readvariableop_dense_424_kernel* 
_output_shapes
:
??*
dtype02<
:model_975/discriminator/dense_424/Tensordot/ReadVariableOp?
0model_975/discriminator/dense_424/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_975/discriminator/dense_424/Tensordot/axes?
0model_975/discriminator/dense_424/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_975/discriminator/dense_424/Tensordot/free?
1model_975/discriminator/dense_424/Tensordot/ShapeShape3model_975/discriminator/dropout_1/Identity:output:0*
T0*
_output_shapes
:23
1model_975/discriminator/dense_424/Tensordot/Shape?
9model_975/discriminator/dense_424/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/discriminator/dense_424/Tensordot/GatherV2/axis?
4model_975/discriminator/dense_424/Tensordot/GatherV2GatherV2:model_975/discriminator/dense_424/Tensordot/Shape:output:09model_975/discriminator/dense_424/Tensordot/free:output:0Bmodel_975/discriminator/dense_424/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_975/discriminator/dense_424/Tensordot/GatherV2?
;model_975/discriminator/dense_424/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_975/discriminator/dense_424/Tensordot/GatherV2_1/axis?
6model_975/discriminator/dense_424/Tensordot/GatherV2_1GatherV2:model_975/discriminator/dense_424/Tensordot/Shape:output:09model_975/discriminator/dense_424/Tensordot/axes:output:0Dmodel_975/discriminator/dense_424/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_975/discriminator/dense_424/Tensordot/GatherV2_1?
1model_975/discriminator/dense_424/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_975/discriminator/dense_424/Tensordot/Const?
0model_975/discriminator/dense_424/Tensordot/ProdProd=model_975/discriminator/dense_424/Tensordot/GatherV2:output:0:model_975/discriminator/dense_424/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_975/discriminator/dense_424/Tensordot/Prod?
3model_975/discriminator/dense_424/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_975/discriminator/dense_424/Tensordot/Const_1?
2model_975/discriminator/dense_424/Tensordot/Prod_1Prod?model_975/discriminator/dense_424/Tensordot/GatherV2_1:output:0<model_975/discriminator/dense_424/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_975/discriminator/dense_424/Tensordot/Prod_1?
7model_975/discriminator/dense_424/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/discriminator/dense_424/Tensordot/concat/axis?
2model_975/discriminator/dense_424/Tensordot/concatConcatV29model_975/discriminator/dense_424/Tensordot/free:output:09model_975/discriminator/dense_424/Tensordot/axes:output:0@model_975/discriminator/dense_424/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_975/discriminator/dense_424/Tensordot/concat?
1model_975/discriminator/dense_424/Tensordot/stackPack9model_975/discriminator/dense_424/Tensordot/Prod:output:0;model_975/discriminator/dense_424/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_975/discriminator/dense_424/Tensordot/stack?
5model_975/discriminator/dense_424/Tensordot/transpose	Transpose3model_975/discriminator/dropout_1/Identity:output:0;model_975/discriminator/dense_424/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_975/discriminator/dense_424/Tensordot/transpose?
3model_975/discriminator/dense_424/Tensordot/ReshapeReshape9model_975/discriminator/dense_424/Tensordot/transpose:y:0:model_975/discriminator/dense_424/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_975/discriminator/dense_424/Tensordot/Reshape?
2model_975/discriminator/dense_424/Tensordot/MatMulMatMul<model_975/discriminator/dense_424/Tensordot/Reshape:output:0Bmodel_975/discriminator/dense_424/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2model_975/discriminator/dense_424/Tensordot/MatMul?
3model_975/discriminator/dense_424/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?25
3model_975/discriminator/dense_424/Tensordot/Const_2?
9model_975/discriminator/dense_424/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/discriminator/dense_424/Tensordot/concat_1/axis?
4model_975/discriminator/dense_424/Tensordot/concat_1ConcatV2=model_975/discriminator/dense_424/Tensordot/GatherV2:output:0<model_975/discriminator/dense_424/Tensordot/Const_2:output:0Bmodel_975/discriminator/dense_424/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_975/discriminator/dense_424/Tensordot/concat_1?
+model_975/discriminator/dense_424/TensordotReshape<model_975/discriminator/dense_424/Tensordot/MatMul:product:0=model_975/discriminator/dense_424/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2-
+model_975/discriminator/dense_424/Tensordot?
8model_975/discriminator/dense_424/BiasAdd/ReadVariableOpReadVariableOpGmodel_975_discriminator_dense_424_biasadd_readvariableop_dense_424_bias*
_output_shapes	
:?*
dtype02:
8model_975/discriminator/dense_424/BiasAdd/ReadVariableOp?
)model_975/discriminator/dense_424/BiasAddBiasAdd4model_975/discriminator/dense_424/Tensordot:output:0@model_975/discriminator/dense_424/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2+
)model_975/discriminator/dense_424/BiasAdd?
&model_975/discriminator/dense_424/TanhTanh2model_975/discriminator/dense_424/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2(
&model_975/discriminator/dense_424/Tanh?
:model_975/discriminator/dense_425/Tensordot/ReadVariableOpReadVariableOpKmodel_975_discriminator_dense_425_tensordot_readvariableop_dense_425_kernel* 
_output_shapes
:
??*
dtype02<
:model_975/discriminator/dense_425/Tensordot/ReadVariableOp?
0model_975/discriminator/dense_425/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_975/discriminator/dense_425/Tensordot/axes?
0model_975/discriminator/dense_425/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_975/discriminator/dense_425/Tensordot/free?
1model_975/discriminator/dense_425/Tensordot/ShapeShape*model_975/discriminator/dense_424/Tanh:y:0*
T0*
_output_shapes
:23
1model_975/discriminator/dense_425/Tensordot/Shape?
9model_975/discriminator/dense_425/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/discriminator/dense_425/Tensordot/GatherV2/axis?
4model_975/discriminator/dense_425/Tensordot/GatherV2GatherV2:model_975/discriminator/dense_425/Tensordot/Shape:output:09model_975/discriminator/dense_425/Tensordot/free:output:0Bmodel_975/discriminator/dense_425/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_975/discriminator/dense_425/Tensordot/GatherV2?
;model_975/discriminator/dense_425/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_975/discriminator/dense_425/Tensordot/GatherV2_1/axis?
6model_975/discriminator/dense_425/Tensordot/GatherV2_1GatherV2:model_975/discriminator/dense_425/Tensordot/Shape:output:09model_975/discriminator/dense_425/Tensordot/axes:output:0Dmodel_975/discriminator/dense_425/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_975/discriminator/dense_425/Tensordot/GatherV2_1?
1model_975/discriminator/dense_425/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_975/discriminator/dense_425/Tensordot/Const?
0model_975/discriminator/dense_425/Tensordot/ProdProd=model_975/discriminator/dense_425/Tensordot/GatherV2:output:0:model_975/discriminator/dense_425/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_975/discriminator/dense_425/Tensordot/Prod?
3model_975/discriminator/dense_425/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_975/discriminator/dense_425/Tensordot/Const_1?
2model_975/discriminator/dense_425/Tensordot/Prod_1Prod?model_975/discriminator/dense_425/Tensordot/GatherV2_1:output:0<model_975/discriminator/dense_425/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_975/discriminator/dense_425/Tensordot/Prod_1?
7model_975/discriminator/dense_425/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/discriminator/dense_425/Tensordot/concat/axis?
2model_975/discriminator/dense_425/Tensordot/concatConcatV29model_975/discriminator/dense_425/Tensordot/free:output:09model_975/discriminator/dense_425/Tensordot/axes:output:0@model_975/discriminator/dense_425/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_975/discriminator/dense_425/Tensordot/concat?
1model_975/discriminator/dense_425/Tensordot/stackPack9model_975/discriminator/dense_425/Tensordot/Prod:output:0;model_975/discriminator/dense_425/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_975/discriminator/dense_425/Tensordot/stack?
5model_975/discriminator/dense_425/Tensordot/transpose	Transpose*model_975/discriminator/dense_424/Tanh:y:0;model_975/discriminator/dense_425/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_975/discriminator/dense_425/Tensordot/transpose?
3model_975/discriminator/dense_425/Tensordot/ReshapeReshape9model_975/discriminator/dense_425/Tensordot/transpose:y:0:model_975/discriminator/dense_425/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_975/discriminator/dense_425/Tensordot/Reshape?
2model_975/discriminator/dense_425/Tensordot/MatMulMatMul<model_975/discriminator/dense_425/Tensordot/Reshape:output:0Bmodel_975/discriminator/dense_425/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2model_975/discriminator/dense_425/Tensordot/MatMul?
3model_975/discriminator/dense_425/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?25
3model_975/discriminator/dense_425/Tensordot/Const_2?
9model_975/discriminator/dense_425/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/discriminator/dense_425/Tensordot/concat_1/axis?
4model_975/discriminator/dense_425/Tensordot/concat_1ConcatV2=model_975/discriminator/dense_425/Tensordot/GatherV2:output:0<model_975/discriminator/dense_425/Tensordot/Const_2:output:0Bmodel_975/discriminator/dense_425/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_975/discriminator/dense_425/Tensordot/concat_1?
+model_975/discriminator/dense_425/TensordotReshape<model_975/discriminator/dense_425/Tensordot/MatMul:product:0=model_975/discriminator/dense_425/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2-
+model_975/discriminator/dense_425/Tensordot?
8model_975/discriminator/dense_425/BiasAdd/ReadVariableOpReadVariableOpGmodel_975_discriminator_dense_425_biasadd_readvariableop_dense_425_bias*
_output_shapes	
:?*
dtype02:
8model_975/discriminator/dense_425/BiasAdd/ReadVariableOp?
)model_975/discriminator/dense_425/BiasAddBiasAdd4model_975/discriminator/dense_425/Tensordot:output:0@model_975/discriminator/dense_425/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2+
)model_975/discriminator/dense_425/BiasAdd?
&model_975/discriminator/dense_425/TanhTanh2model_975/discriminator/dense_425/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2(
&model_975/discriminator/dense_425/Tanh?
:model_975/discriminator/dense_426/Tensordot/ReadVariableOpReadVariableOpKmodel_975_discriminator_dense_426_tensordot_readvariableop_dense_426_kernel* 
_output_shapes
:
??*
dtype02<
:model_975/discriminator/dense_426/Tensordot/ReadVariableOp?
0model_975/discriminator/dense_426/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_975/discriminator/dense_426/Tensordot/axes?
0model_975/discriminator/dense_426/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_975/discriminator/dense_426/Tensordot/free?
1model_975/discriminator/dense_426/Tensordot/ShapeShape*model_975/discriminator/dense_425/Tanh:y:0*
T0*
_output_shapes
:23
1model_975/discriminator/dense_426/Tensordot/Shape?
9model_975/discriminator/dense_426/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/discriminator/dense_426/Tensordot/GatherV2/axis?
4model_975/discriminator/dense_426/Tensordot/GatherV2GatherV2:model_975/discriminator/dense_426/Tensordot/Shape:output:09model_975/discriminator/dense_426/Tensordot/free:output:0Bmodel_975/discriminator/dense_426/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_975/discriminator/dense_426/Tensordot/GatherV2?
;model_975/discriminator/dense_426/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_975/discriminator/dense_426/Tensordot/GatherV2_1/axis?
6model_975/discriminator/dense_426/Tensordot/GatherV2_1GatherV2:model_975/discriminator/dense_426/Tensordot/Shape:output:09model_975/discriminator/dense_426/Tensordot/axes:output:0Dmodel_975/discriminator/dense_426/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_975/discriminator/dense_426/Tensordot/GatherV2_1?
1model_975/discriminator/dense_426/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_975/discriminator/dense_426/Tensordot/Const?
0model_975/discriminator/dense_426/Tensordot/ProdProd=model_975/discriminator/dense_426/Tensordot/GatherV2:output:0:model_975/discriminator/dense_426/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_975/discriminator/dense_426/Tensordot/Prod?
3model_975/discriminator/dense_426/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_975/discriminator/dense_426/Tensordot/Const_1?
2model_975/discriminator/dense_426/Tensordot/Prod_1Prod?model_975/discriminator/dense_426/Tensordot/GatherV2_1:output:0<model_975/discriminator/dense_426/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_975/discriminator/dense_426/Tensordot/Prod_1?
7model_975/discriminator/dense_426/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/discriminator/dense_426/Tensordot/concat/axis?
2model_975/discriminator/dense_426/Tensordot/concatConcatV29model_975/discriminator/dense_426/Tensordot/free:output:09model_975/discriminator/dense_426/Tensordot/axes:output:0@model_975/discriminator/dense_426/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_975/discriminator/dense_426/Tensordot/concat?
1model_975/discriminator/dense_426/Tensordot/stackPack9model_975/discriminator/dense_426/Tensordot/Prod:output:0;model_975/discriminator/dense_426/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_975/discriminator/dense_426/Tensordot/stack?
5model_975/discriminator/dense_426/Tensordot/transpose	Transpose*model_975/discriminator/dense_425/Tanh:y:0;model_975/discriminator/dense_426/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_975/discriminator/dense_426/Tensordot/transpose?
3model_975/discriminator/dense_426/Tensordot/ReshapeReshape9model_975/discriminator/dense_426/Tensordot/transpose:y:0:model_975/discriminator/dense_426/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_975/discriminator/dense_426/Tensordot/Reshape?
2model_975/discriminator/dense_426/Tensordot/MatMulMatMul<model_975/discriminator/dense_426/Tensordot/Reshape:output:0Bmodel_975/discriminator/dense_426/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2model_975/discriminator/dense_426/Tensordot/MatMul?
3model_975/discriminator/dense_426/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?25
3model_975/discriminator/dense_426/Tensordot/Const_2?
9model_975/discriminator/dense_426/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/discriminator/dense_426/Tensordot/concat_1/axis?
4model_975/discriminator/dense_426/Tensordot/concat_1ConcatV2=model_975/discriminator/dense_426/Tensordot/GatherV2:output:0<model_975/discriminator/dense_426/Tensordot/Const_2:output:0Bmodel_975/discriminator/dense_426/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_975/discriminator/dense_426/Tensordot/concat_1?
+model_975/discriminator/dense_426/TensordotReshape<model_975/discriminator/dense_426/Tensordot/MatMul:product:0=model_975/discriminator/dense_426/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2-
+model_975/discriminator/dense_426/Tensordot?
8model_975/discriminator/dense_426/BiasAdd/ReadVariableOpReadVariableOpGmodel_975_discriminator_dense_426_biasadd_readvariableop_dense_426_bias*
_output_shapes	
:?*
dtype02:
8model_975/discriminator/dense_426/BiasAdd/ReadVariableOp?
)model_975/discriminator/dense_426/BiasAddBiasAdd4model_975/discriminator/dense_426/Tensordot:output:0@model_975/discriminator/dense_426/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2+
)model_975/discriminator/dense_426/BiasAdd?
&model_975/discriminator/dense_426/TanhTanh2model_975/discriminator/dense_426/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2(
&model_975/discriminator/dense_426/Tanh?
:model_975/discriminator/dense_427/Tensordot/ReadVariableOpReadVariableOpKmodel_975_discriminator_dense_427_tensordot_readvariableop_dense_427_kernel*
_output_shapes
:	?*
dtype02<
:model_975/discriminator/dense_427/Tensordot/ReadVariableOp?
0model_975/discriminator/dense_427/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_975/discriminator/dense_427/Tensordot/axes?
0model_975/discriminator/dense_427/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_975/discriminator/dense_427/Tensordot/free?
1model_975/discriminator/dense_427/Tensordot/ShapeShape*model_975/discriminator/dense_426/Tanh:y:0*
T0*
_output_shapes
:23
1model_975/discriminator/dense_427/Tensordot/Shape?
9model_975/discriminator/dense_427/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/discriminator/dense_427/Tensordot/GatherV2/axis?
4model_975/discriminator/dense_427/Tensordot/GatherV2GatherV2:model_975/discriminator/dense_427/Tensordot/Shape:output:09model_975/discriminator/dense_427/Tensordot/free:output:0Bmodel_975/discriminator/dense_427/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_975/discriminator/dense_427/Tensordot/GatherV2?
;model_975/discriminator/dense_427/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_975/discriminator/dense_427/Tensordot/GatherV2_1/axis?
6model_975/discriminator/dense_427/Tensordot/GatherV2_1GatherV2:model_975/discriminator/dense_427/Tensordot/Shape:output:09model_975/discriminator/dense_427/Tensordot/axes:output:0Dmodel_975/discriminator/dense_427/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_975/discriminator/dense_427/Tensordot/GatherV2_1?
1model_975/discriminator/dense_427/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_975/discriminator/dense_427/Tensordot/Const?
0model_975/discriminator/dense_427/Tensordot/ProdProd=model_975/discriminator/dense_427/Tensordot/GatherV2:output:0:model_975/discriminator/dense_427/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_975/discriminator/dense_427/Tensordot/Prod?
3model_975/discriminator/dense_427/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_975/discriminator/dense_427/Tensordot/Const_1?
2model_975/discriminator/dense_427/Tensordot/Prod_1Prod?model_975/discriminator/dense_427/Tensordot/GatherV2_1:output:0<model_975/discriminator/dense_427/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_975/discriminator/dense_427/Tensordot/Prod_1?
7model_975/discriminator/dense_427/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_975/discriminator/dense_427/Tensordot/concat/axis?
2model_975/discriminator/dense_427/Tensordot/concatConcatV29model_975/discriminator/dense_427/Tensordot/free:output:09model_975/discriminator/dense_427/Tensordot/axes:output:0@model_975/discriminator/dense_427/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_975/discriminator/dense_427/Tensordot/concat?
1model_975/discriminator/dense_427/Tensordot/stackPack9model_975/discriminator/dense_427/Tensordot/Prod:output:0;model_975/discriminator/dense_427/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_975/discriminator/dense_427/Tensordot/stack?
5model_975/discriminator/dense_427/Tensordot/transpose	Transpose*model_975/discriminator/dense_426/Tanh:y:0;model_975/discriminator/dense_427/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_975/discriminator/dense_427/Tensordot/transpose?
3model_975/discriminator/dense_427/Tensordot/ReshapeReshape9model_975/discriminator/dense_427/Tensordot/transpose:y:0:model_975/discriminator/dense_427/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_975/discriminator/dense_427/Tensordot/Reshape?
2model_975/discriminator/dense_427/Tensordot/MatMulMatMul<model_975/discriminator/dense_427/Tensordot/Reshape:output:0Bmodel_975/discriminator/dense_427/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
2model_975/discriminator/dense_427/Tensordot/MatMul?
3model_975/discriminator/dense_427/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_975/discriminator/dense_427/Tensordot/Const_2?
9model_975/discriminator/dense_427/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_975/discriminator/dense_427/Tensordot/concat_1/axis?
4model_975/discriminator/dense_427/Tensordot/concat_1ConcatV2=model_975/discriminator/dense_427/Tensordot/GatherV2:output:0<model_975/discriminator/dense_427/Tensordot/Const_2:output:0Bmodel_975/discriminator/dense_427/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_975/discriminator/dense_427/Tensordot/concat_1?
+model_975/discriminator/dense_427/TensordotReshape<model_975/discriminator/dense_427/Tensordot/MatMul:product:0=model_975/discriminator/dense_427/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2-
+model_975/discriminator/dense_427/Tensordot?
8model_975/discriminator/dense_427/BiasAdd/ReadVariableOpReadVariableOpGmodel_975_discriminator_dense_427_biasadd_readvariableop_dense_427_bias*
_output_shapes
:*
dtype02:
8model_975/discriminator/dense_427/BiasAdd/ReadVariableOp?
)model_975/discriminator/dense_427/BiasAddBiasAdd4model_975/discriminator/dense_427/Tensordot:output:0@model_975/discriminator/dense_427/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2+
)model_975/discriminator/dense_427/BiasAdd?
&model_975/discriminator/dense_427/TanhTanh2model_975/discriminator/dense_427/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2(
&model_975/discriminator/dense_427/Tanh?
(model_975/discriminator/flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2*
(model_975/discriminator/flatten_52/Const?
*model_975/discriminator/flatten_52/ReshapeReshape*model_975/discriminator/dense_427/Tanh:y:01model_975/discriminator/flatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2,
*model_975/discriminator/flatten_52/Reshape?
7model_975/discriminator/dense_428/MatMul/ReadVariableOpReadVariableOpHmodel_975_discriminator_dense_428_matmul_readvariableop_dense_428_kernel*
_output_shapes

:*
dtype029
7model_975/discriminator/dense_428/MatMul/ReadVariableOp?
(model_975/discriminator/dense_428/MatMulMatMul3model_975/discriminator/flatten_52/Reshape:output:0?model_975/discriminator/dense_428/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(model_975/discriminator/dense_428/MatMul?
8model_975/discriminator/dense_428/BiasAdd/ReadVariableOpReadVariableOpGmodel_975_discriminator_dense_428_biasadd_readvariableop_dense_428_bias*
_output_shapes
:*
dtype02:
8model_975/discriminator/dense_428/BiasAdd/ReadVariableOp?
)model_975/discriminator/dense_428/BiasAddBiasAdd2model_975/discriminator/dense_428/MatMul:product:0@model_975/discriminator/dense_428/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)model_975/discriminator/dense_428/BiasAdd?
)model_975/discriminator/dense_428/SigmoidSigmoid2model_975/discriminator/dense_428/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2+
)model_975/discriminator/dense_428/Sigmoid?
IdentityIdentity-model_975/discriminator/dense_428/Sigmoid:y:07^model_975/autoencoder/dense_421/BiasAdd/ReadVariableOp9^model_975/autoencoder/dense_421/Tensordot/ReadVariableOp7^model_975/autoencoder/dense_422/BiasAdd/ReadVariableOp9^model_975/autoencoder/dense_422/Tensordot/ReadVariableOp7^model_975/autoencoder/dense_423/BiasAdd/ReadVariableOp9^model_975/autoencoder/dense_423/Tensordot/ReadVariableOp9^model_975/discriminator/dense_424/BiasAdd/ReadVariableOp;^model_975/discriminator/dense_424/Tensordot/ReadVariableOp9^model_975/discriminator/dense_425/BiasAdd/ReadVariableOp;^model_975/discriminator/dense_425/Tensordot/ReadVariableOp9^model_975/discriminator/dense_426/BiasAdd/ReadVariableOp;^model_975/discriminator/dense_426/Tensordot/ReadVariableOp9^model_975/discriminator/dense_427/BiasAdd/ReadVariableOp;^model_975/discriminator/dense_427/Tensordot/ReadVariableOp9^model_975/discriminator/dense_428/BiasAdd/ReadVariableOp8^model_975/discriminator/dense_428/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::2p
6model_975/autoencoder/dense_421/BiasAdd/ReadVariableOp6model_975/autoencoder/dense_421/BiasAdd/ReadVariableOp2t
8model_975/autoencoder/dense_421/Tensordot/ReadVariableOp8model_975/autoencoder/dense_421/Tensordot/ReadVariableOp2p
6model_975/autoencoder/dense_422/BiasAdd/ReadVariableOp6model_975/autoencoder/dense_422/BiasAdd/ReadVariableOp2t
8model_975/autoencoder/dense_422/Tensordot/ReadVariableOp8model_975/autoencoder/dense_422/Tensordot/ReadVariableOp2p
6model_975/autoencoder/dense_423/BiasAdd/ReadVariableOp6model_975/autoencoder/dense_423/BiasAdd/ReadVariableOp2t
8model_975/autoencoder/dense_423/Tensordot/ReadVariableOp8model_975/autoencoder/dense_423/Tensordot/ReadVariableOp2t
8model_975/discriminator/dense_424/BiasAdd/ReadVariableOp8model_975/discriminator/dense_424/BiasAdd/ReadVariableOp2x
:model_975/discriminator/dense_424/Tensordot/ReadVariableOp:model_975/discriminator/dense_424/Tensordot/ReadVariableOp2t
8model_975/discriminator/dense_425/BiasAdd/ReadVariableOp8model_975/discriminator/dense_425/BiasAdd/ReadVariableOp2x
:model_975/discriminator/dense_425/Tensordot/ReadVariableOp:model_975/discriminator/dense_425/Tensordot/ReadVariableOp2t
8model_975/discriminator/dense_426/BiasAdd/ReadVariableOp8model_975/discriminator/dense_426/BiasAdd/ReadVariableOp2x
:model_975/discriminator/dense_426/Tensordot/ReadVariableOp:model_975/discriminator/dense_426/Tensordot/ReadVariableOp2t
8model_975/discriminator/dense_427/BiasAdd/ReadVariableOp8model_975/discriminator/dense_427/BiasAdd/ReadVariableOp2x
:model_975/discriminator/dense_427/Tensordot/ReadVariableOp:model_975/discriminator/dense_427/Tensordot/ReadVariableOp2t
8model_975/discriminator/dense_428/BiasAdd/ReadVariableOp8model_975/discriminator/dense_428/BiasAdd/ReadVariableOp2r
7model_975/discriminator/dense_428/MatMul/ReadVariableOp7model_975/discriminator/dense_428/MatMul/ReadVariableOp:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
??
?
K__inference_discriminator_layer_call_and_return_conditional_losses_61000876

inputs7
3dense_424_tensordot_readvariableop_dense_424_kernel3
/dense_424_biasadd_readvariableop_dense_424_bias7
3dense_425_tensordot_readvariableop_dense_425_kernel3
/dense_425_biasadd_readvariableop_dense_425_bias7
3dense_426_tensordot_readvariableop_dense_426_kernel3
/dense_426_biasadd_readvariableop_dense_426_bias7
3dense_427_tensordot_readvariableop_dense_427_kernel3
/dense_427_biasadd_readvariableop_dense_427_bias4
0dense_428_matmul_readvariableop_dense_428_kernel3
/dense_428_biasadd_readvariableop_dense_428_bias
identity?? dense_424/BiasAdd/ReadVariableOp?"dense_424/Tensordot/ReadVariableOp? dense_425/BiasAdd/ReadVariableOp?"dense_425/Tensordot/ReadVariableOp? dense_426/BiasAdd/ReadVariableOp?"dense_426/Tensordot/ReadVariableOp? dense_427/BiasAdd/ReadVariableOp?"dense_427/Tensordot/ReadVariableOp? dense_428/BiasAdd/ReadVariableOp?dense_428/MatMul/ReadVariableOps
dropout_1/IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2
dropout_1/Identity?
"dense_424/Tensordot/ReadVariableOpReadVariableOp3dense_424_tensordot_readvariableop_dense_424_kernel* 
_output_shapes
:
??*
dtype02$
"dense_424/Tensordot/ReadVariableOp~
dense_424/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_424/Tensordot/axes?
dense_424/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_424/Tensordot/free?
dense_424/Tensordot/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:2
dense_424/Tensordot/Shape?
!dense_424/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_424/Tensordot/GatherV2/axis?
dense_424/Tensordot/GatherV2GatherV2"dense_424/Tensordot/Shape:output:0!dense_424/Tensordot/free:output:0*dense_424/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_424/Tensordot/GatherV2?
#dense_424/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_424/Tensordot/GatherV2_1/axis?
dense_424/Tensordot/GatherV2_1GatherV2"dense_424/Tensordot/Shape:output:0!dense_424/Tensordot/axes:output:0,dense_424/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_424/Tensordot/GatherV2_1?
dense_424/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_424/Tensordot/Const?
dense_424/Tensordot/ProdProd%dense_424/Tensordot/GatherV2:output:0"dense_424/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_424/Tensordot/Prod?
dense_424/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_424/Tensordot/Const_1?
dense_424/Tensordot/Prod_1Prod'dense_424/Tensordot/GatherV2_1:output:0$dense_424/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_424/Tensordot/Prod_1?
dense_424/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_424/Tensordot/concat/axis?
dense_424/Tensordot/concatConcatV2!dense_424/Tensordot/free:output:0!dense_424/Tensordot/axes:output:0(dense_424/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_424/Tensordot/concat?
dense_424/Tensordot/stackPack!dense_424/Tensordot/Prod:output:0#dense_424/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_424/Tensordot/stack?
dense_424/Tensordot/transpose	Transposedropout_1/Identity:output:0#dense_424/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_424/Tensordot/transpose?
dense_424/Tensordot/ReshapeReshape!dense_424/Tensordot/transpose:y:0"dense_424/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_424/Tensordot/Reshape?
dense_424/Tensordot/MatMulMatMul$dense_424/Tensordot/Reshape:output:0*dense_424/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_424/Tensordot/MatMul?
dense_424/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_424/Tensordot/Const_2?
!dense_424/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_424/Tensordot/concat_1/axis?
dense_424/Tensordot/concat_1ConcatV2%dense_424/Tensordot/GatherV2:output:0$dense_424/Tensordot/Const_2:output:0*dense_424/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_424/Tensordot/concat_1?
dense_424/TensordotReshape$dense_424/Tensordot/MatMul:product:0%dense_424/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_424/Tensordot?
 dense_424/BiasAdd/ReadVariableOpReadVariableOp/dense_424_biasadd_readvariableop_dense_424_bias*
_output_shapes	
:?*
dtype02"
 dense_424/BiasAdd/ReadVariableOp?
dense_424/BiasAddBiasAdddense_424/Tensordot:output:0(dense_424/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_424/BiasAdd{
dense_424/TanhTanhdense_424/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_424/Tanh?
"dense_425/Tensordot/ReadVariableOpReadVariableOp3dense_425_tensordot_readvariableop_dense_425_kernel* 
_output_shapes
:
??*
dtype02$
"dense_425/Tensordot/ReadVariableOp~
dense_425/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_425/Tensordot/axes?
dense_425/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_425/Tensordot/freex
dense_425/Tensordot/ShapeShapedense_424/Tanh:y:0*
T0*
_output_shapes
:2
dense_425/Tensordot/Shape?
!dense_425/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_425/Tensordot/GatherV2/axis?
dense_425/Tensordot/GatherV2GatherV2"dense_425/Tensordot/Shape:output:0!dense_425/Tensordot/free:output:0*dense_425/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_425/Tensordot/GatherV2?
#dense_425/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_425/Tensordot/GatherV2_1/axis?
dense_425/Tensordot/GatherV2_1GatherV2"dense_425/Tensordot/Shape:output:0!dense_425/Tensordot/axes:output:0,dense_425/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_425/Tensordot/GatherV2_1?
dense_425/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_425/Tensordot/Const?
dense_425/Tensordot/ProdProd%dense_425/Tensordot/GatherV2:output:0"dense_425/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_425/Tensordot/Prod?
dense_425/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_425/Tensordot/Const_1?
dense_425/Tensordot/Prod_1Prod'dense_425/Tensordot/GatherV2_1:output:0$dense_425/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_425/Tensordot/Prod_1?
dense_425/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_425/Tensordot/concat/axis?
dense_425/Tensordot/concatConcatV2!dense_425/Tensordot/free:output:0!dense_425/Tensordot/axes:output:0(dense_425/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_425/Tensordot/concat?
dense_425/Tensordot/stackPack!dense_425/Tensordot/Prod:output:0#dense_425/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_425/Tensordot/stack?
dense_425/Tensordot/transpose	Transposedense_424/Tanh:y:0#dense_425/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_425/Tensordot/transpose?
dense_425/Tensordot/ReshapeReshape!dense_425/Tensordot/transpose:y:0"dense_425/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_425/Tensordot/Reshape?
dense_425/Tensordot/MatMulMatMul$dense_425/Tensordot/Reshape:output:0*dense_425/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_425/Tensordot/MatMul?
dense_425/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_425/Tensordot/Const_2?
!dense_425/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_425/Tensordot/concat_1/axis?
dense_425/Tensordot/concat_1ConcatV2%dense_425/Tensordot/GatherV2:output:0$dense_425/Tensordot/Const_2:output:0*dense_425/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_425/Tensordot/concat_1?
dense_425/TensordotReshape$dense_425/Tensordot/MatMul:product:0%dense_425/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_425/Tensordot?
 dense_425/BiasAdd/ReadVariableOpReadVariableOp/dense_425_biasadd_readvariableop_dense_425_bias*
_output_shapes	
:?*
dtype02"
 dense_425/BiasAdd/ReadVariableOp?
dense_425/BiasAddBiasAdddense_425/Tensordot:output:0(dense_425/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_425/BiasAdd{
dense_425/TanhTanhdense_425/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_425/Tanh?
"dense_426/Tensordot/ReadVariableOpReadVariableOp3dense_426_tensordot_readvariableop_dense_426_kernel* 
_output_shapes
:
??*
dtype02$
"dense_426/Tensordot/ReadVariableOp~
dense_426/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_426/Tensordot/axes?
dense_426/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_426/Tensordot/freex
dense_426/Tensordot/ShapeShapedense_425/Tanh:y:0*
T0*
_output_shapes
:2
dense_426/Tensordot/Shape?
!dense_426/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_426/Tensordot/GatherV2/axis?
dense_426/Tensordot/GatherV2GatherV2"dense_426/Tensordot/Shape:output:0!dense_426/Tensordot/free:output:0*dense_426/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_426/Tensordot/GatherV2?
#dense_426/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_426/Tensordot/GatherV2_1/axis?
dense_426/Tensordot/GatherV2_1GatherV2"dense_426/Tensordot/Shape:output:0!dense_426/Tensordot/axes:output:0,dense_426/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_426/Tensordot/GatherV2_1?
dense_426/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_426/Tensordot/Const?
dense_426/Tensordot/ProdProd%dense_426/Tensordot/GatherV2:output:0"dense_426/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_426/Tensordot/Prod?
dense_426/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_426/Tensordot/Const_1?
dense_426/Tensordot/Prod_1Prod'dense_426/Tensordot/GatherV2_1:output:0$dense_426/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_426/Tensordot/Prod_1?
dense_426/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_426/Tensordot/concat/axis?
dense_426/Tensordot/concatConcatV2!dense_426/Tensordot/free:output:0!dense_426/Tensordot/axes:output:0(dense_426/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_426/Tensordot/concat?
dense_426/Tensordot/stackPack!dense_426/Tensordot/Prod:output:0#dense_426/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_426/Tensordot/stack?
dense_426/Tensordot/transpose	Transposedense_425/Tanh:y:0#dense_426/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_426/Tensordot/transpose?
dense_426/Tensordot/ReshapeReshape!dense_426/Tensordot/transpose:y:0"dense_426/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_426/Tensordot/Reshape?
dense_426/Tensordot/MatMulMatMul$dense_426/Tensordot/Reshape:output:0*dense_426/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_426/Tensordot/MatMul?
dense_426/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_426/Tensordot/Const_2?
!dense_426/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_426/Tensordot/concat_1/axis?
dense_426/Tensordot/concat_1ConcatV2%dense_426/Tensordot/GatherV2:output:0$dense_426/Tensordot/Const_2:output:0*dense_426/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_426/Tensordot/concat_1?
dense_426/TensordotReshape$dense_426/Tensordot/MatMul:product:0%dense_426/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_426/Tensordot?
 dense_426/BiasAdd/ReadVariableOpReadVariableOp/dense_426_biasadd_readvariableop_dense_426_bias*
_output_shapes	
:?*
dtype02"
 dense_426/BiasAdd/ReadVariableOp?
dense_426/BiasAddBiasAdddense_426/Tensordot:output:0(dense_426/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_426/BiasAdd{
dense_426/TanhTanhdense_426/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_426/Tanh?
"dense_427/Tensordot/ReadVariableOpReadVariableOp3dense_427_tensordot_readvariableop_dense_427_kernel*
_output_shapes
:	?*
dtype02$
"dense_427/Tensordot/ReadVariableOp~
dense_427/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_427/Tensordot/axes?
dense_427/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_427/Tensordot/freex
dense_427/Tensordot/ShapeShapedense_426/Tanh:y:0*
T0*
_output_shapes
:2
dense_427/Tensordot/Shape?
!dense_427/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_427/Tensordot/GatherV2/axis?
dense_427/Tensordot/GatherV2GatherV2"dense_427/Tensordot/Shape:output:0!dense_427/Tensordot/free:output:0*dense_427/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_427/Tensordot/GatherV2?
#dense_427/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_427/Tensordot/GatherV2_1/axis?
dense_427/Tensordot/GatherV2_1GatherV2"dense_427/Tensordot/Shape:output:0!dense_427/Tensordot/axes:output:0,dense_427/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_427/Tensordot/GatherV2_1?
dense_427/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_427/Tensordot/Const?
dense_427/Tensordot/ProdProd%dense_427/Tensordot/GatherV2:output:0"dense_427/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_427/Tensordot/Prod?
dense_427/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_427/Tensordot/Const_1?
dense_427/Tensordot/Prod_1Prod'dense_427/Tensordot/GatherV2_1:output:0$dense_427/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_427/Tensordot/Prod_1?
dense_427/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_427/Tensordot/concat/axis?
dense_427/Tensordot/concatConcatV2!dense_427/Tensordot/free:output:0!dense_427/Tensordot/axes:output:0(dense_427/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_427/Tensordot/concat?
dense_427/Tensordot/stackPack!dense_427/Tensordot/Prod:output:0#dense_427/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_427/Tensordot/stack?
dense_427/Tensordot/transpose	Transposedense_426/Tanh:y:0#dense_427/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_427/Tensordot/transpose?
dense_427/Tensordot/ReshapeReshape!dense_427/Tensordot/transpose:y:0"dense_427/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_427/Tensordot/Reshape?
dense_427/Tensordot/MatMulMatMul$dense_427/Tensordot/Reshape:output:0*dense_427/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_427/Tensordot/MatMul?
dense_427/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_427/Tensordot/Const_2?
!dense_427/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_427/Tensordot/concat_1/axis?
dense_427/Tensordot/concat_1ConcatV2%dense_427/Tensordot/GatherV2:output:0$dense_427/Tensordot/Const_2:output:0*dense_427/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_427/Tensordot/concat_1?
dense_427/TensordotReshape$dense_427/Tensordot/MatMul:product:0%dense_427/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_427/Tensordot?
 dense_427/BiasAdd/ReadVariableOpReadVariableOp/dense_427_biasadd_readvariableop_dense_427_bias*
_output_shapes
:*
dtype02"
 dense_427/BiasAdd/ReadVariableOp?
dense_427/BiasAddBiasAdddense_427/Tensordot:output:0(dense_427/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_427/BiasAddz
dense_427/TanhTanhdense_427/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense_427/Tanhu
flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_52/Const?
flatten_52/ReshapeReshapedense_427/Tanh:y:0flatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_52/Reshape?
dense_428/MatMul/ReadVariableOpReadVariableOp0dense_428_matmul_readvariableop_dense_428_kernel*
_output_shapes

:*
dtype02!
dense_428/MatMul/ReadVariableOp?
dense_428/MatMulMatMulflatten_52/Reshape:output:0'dense_428/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_428/MatMul?
 dense_428/BiasAdd/ReadVariableOpReadVariableOp/dense_428_biasadd_readvariableop_dense_428_bias*
_output_shapes
:*
dtype02"
 dense_428/BiasAdd/ReadVariableOp?
dense_428/BiasAddBiasAdddense_428/MatMul:product:0(dense_428/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_428/BiasAdd
dense_428/SigmoidSigmoiddense_428/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_428/Sigmoid?
IdentityIdentitydense_428/Sigmoid:y:0!^dense_424/BiasAdd/ReadVariableOp#^dense_424/Tensordot/ReadVariableOp!^dense_425/BiasAdd/ReadVariableOp#^dense_425/Tensordot/ReadVariableOp!^dense_426/BiasAdd/ReadVariableOp#^dense_426/Tensordot/ReadVariableOp!^dense_427/BiasAdd/ReadVariableOp#^dense_427/Tensordot/ReadVariableOp!^dense_428/BiasAdd/ReadVariableOp ^dense_428/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::2D
 dense_424/BiasAdd/ReadVariableOp dense_424/BiasAdd/ReadVariableOp2H
"dense_424/Tensordot/ReadVariableOp"dense_424/Tensordot/ReadVariableOp2D
 dense_425/BiasAdd/ReadVariableOp dense_425/BiasAdd/ReadVariableOp2H
"dense_425/Tensordot/ReadVariableOp"dense_425/Tensordot/ReadVariableOp2D
 dense_426/BiasAdd/ReadVariableOp dense_426/BiasAdd/ReadVariableOp2H
"dense_426/Tensordot/ReadVariableOp"dense_426/Tensordot/ReadVariableOp2D
 dense_427/BiasAdd/ReadVariableOp dense_427/BiasAdd/ReadVariableOp2H
"dense_427/Tensordot/ReadVariableOp"dense_427/Tensordot/ReadVariableOp2D
 dense_428/BiasAdd/ReadVariableOp dense_428/BiasAdd/ReadVariableOp2B
dense_428/MatMul/ReadVariableOpdense_428/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
0__inference_discriminator_layer_call_fn_61000906

inputs
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609993962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_426_layer_call_and_return_conditional_losses_61001165

inputs-
)tensordot_readvariableop_dense_426_kernel)
%biasadd_readvariableop_dense_426_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_426_kernel* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_426_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_model_975_layer_call_fn_60999763
input_13
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13dense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_biasdense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_975_layer_call_and_return_conditional_losses_609997442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
?
?
G__inference_dense_423_layer_call_and_return_conditional_losses_61001024

inputs-
)tensordot_readvariableop_dense_423_kernel)
%biasadd_readvariableop_dense_423_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_60999786
input_13
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13dense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_biasdense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_609988582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
?q
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000326

inputs7
3dense_421_tensordot_readvariableop_dense_421_kernel3
/dense_421_biasadd_readvariableop_dense_421_bias7
3dense_422_tensordot_readvariableop_dense_422_kernel3
/dense_422_biasadd_readvariableop_dense_422_bias7
3dense_423_tensordot_readvariableop_dense_423_kernel3
/dense_423_biasadd_readvariableop_dense_423_bias
identity?? dense_421/BiasAdd/ReadVariableOp?"dense_421/Tensordot/ReadVariableOp? dense_422/BiasAdd/ReadVariableOp?"dense_422/Tensordot/ReadVariableOp? dense_423/BiasAdd/ReadVariableOp?"dense_423/Tensordot/ReadVariableOp?
"dense_421/Tensordot/ReadVariableOpReadVariableOp3dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02$
"dense_421/Tensordot/ReadVariableOp~
dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_421/Tensordot/axes?
dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_421/Tensordot/freel
dense_421/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_421/Tensordot/Shape?
!dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/GatherV2/axis?
dense_421/Tensordot/GatherV2GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/free:output:0*dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_421/Tensordot/GatherV2?
#dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_421/Tensordot/GatherV2_1/axis?
dense_421/Tensordot/GatherV2_1GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/axes:output:0,dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_421/Tensordot/GatherV2_1?
dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const?
dense_421/Tensordot/ProdProd%dense_421/Tensordot/GatherV2:output:0"dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod?
dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_1?
dense_421/Tensordot/Prod_1Prod'dense_421/Tensordot/GatherV2_1:output:0$dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod_1?
dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_421/Tensordot/concat/axis?
dense_421/Tensordot/concatConcatV2!dense_421/Tensordot/free:output:0!dense_421/Tensordot/axes:output:0(dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat?
dense_421/Tensordot/stackPack!dense_421/Tensordot/Prod:output:0#dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/stack?
dense_421/Tensordot/transpose	Transposeinputs#dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_421/Tensordot/transpose?
dense_421/Tensordot/ReshapeReshape!dense_421/Tensordot/transpose:y:0"dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_421/Tensordot/Reshape?
dense_421/Tensordot/MatMulMatMul$dense_421/Tensordot/Reshape:output:0*dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_421/Tensordot/MatMul?
dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_2?
!dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/concat_1/axis?
dense_421/Tensordot/concat_1ConcatV2%dense_421/Tensordot/GatherV2:output:0$dense_421/Tensordot/Const_2:output:0*dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat_1?
dense_421/TensordotReshape$dense_421/Tensordot/MatMul:product:0%dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tensordot?
 dense_421/BiasAdd/ReadVariableOpReadVariableOp/dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02"
 dense_421/BiasAdd/ReadVariableOp?
dense_421/BiasAddBiasAdddense_421/Tensordot:output:0(dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_421/BiasAddz
dense_421/TanhTanhdense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tanh?
"dense_422/Tensordot/ReadVariableOpReadVariableOp3dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_422/Tensordot/ReadVariableOp~
dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_422/Tensordot/axes?
dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_422/Tensordot/freex
dense_422/Tensordot/ShapeShapedense_421/Tanh:y:0*
T0*
_output_shapes
:2
dense_422/Tensordot/Shape?
!dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/GatherV2/axis?
dense_422/Tensordot/GatherV2GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/free:output:0*dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_422/Tensordot/GatherV2?
#dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_422/Tensordot/GatherV2_1/axis?
dense_422/Tensordot/GatherV2_1GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/axes:output:0,dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_422/Tensordot/GatherV2_1?
dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const?
dense_422/Tensordot/ProdProd%dense_422/Tensordot/GatherV2:output:0"dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod?
dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const_1?
dense_422/Tensordot/Prod_1Prod'dense_422/Tensordot/GatherV2_1:output:0$dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod_1?
dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_422/Tensordot/concat/axis?
dense_422/Tensordot/concatConcatV2!dense_422/Tensordot/free:output:0!dense_422/Tensordot/axes:output:0(dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat?
dense_422/Tensordot/stackPack!dense_422/Tensordot/Prod:output:0#dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/stack?
dense_422/Tensordot/transpose	Transposedense_421/Tanh:y:0#dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_422/Tensordot/transpose?
dense_422/Tensordot/ReshapeReshape!dense_422/Tensordot/transpose:y:0"dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_422/Tensordot/Reshape?
dense_422/Tensordot/MatMulMatMul$dense_422/Tensordot/Reshape:output:0*dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_422/Tensordot/MatMul?
dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_422/Tensordot/Const_2?
!dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/concat_1/axis?
dense_422/Tensordot/concat_1ConcatV2%dense_422/Tensordot/GatherV2:output:0$dense_422/Tensordot/Const_2:output:0*dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat_1?
dense_422/TensordotReshape$dense_422/Tensordot/MatMul:product:0%dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tensordot?
 dense_422/BiasAdd/ReadVariableOpReadVariableOp/dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02"
 dense_422/BiasAdd/ReadVariableOp?
dense_422/BiasAddBiasAdddense_422/Tensordot:output:0(dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_422/BiasAdd{
dense_422/TanhTanhdense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tanht

add_53/addAddV2inputsdense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2

add_53/add?
"dense_423/Tensordot/ReadVariableOpReadVariableOp3dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02$
"dense_423/Tensordot/ReadVariableOp~
dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_423/Tensordot/axes?
dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_423/Tensordot/freet
dense_423/Tensordot/ShapeShapeadd_53/add:z:0*
T0*
_output_shapes
:2
dense_423/Tensordot/Shape?
!dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/GatherV2/axis?
dense_423/Tensordot/GatherV2GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/free:output:0*dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_423/Tensordot/GatherV2?
#dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_423/Tensordot/GatherV2_1/axis?
dense_423/Tensordot/GatherV2_1GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/axes:output:0,dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_423/Tensordot/GatherV2_1?
dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const?
dense_423/Tensordot/ProdProd%dense_423/Tensordot/GatherV2:output:0"dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod?
dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const_1?
dense_423/Tensordot/Prod_1Prod'dense_423/Tensordot/GatherV2_1:output:0$dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod_1?
dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_423/Tensordot/concat/axis?
dense_423/Tensordot/concatConcatV2!dense_423/Tensordot/free:output:0!dense_423/Tensordot/axes:output:0(dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat?
dense_423/Tensordot/stackPack!dense_423/Tensordot/Prod:output:0#dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/stack?
dense_423/Tensordot/transpose	Transposeadd_53/add:z:0#dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot/transpose?
dense_423/Tensordot/ReshapeReshape!dense_423/Tensordot/transpose:y:0"dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_423/Tensordot/Reshape?
dense_423/Tensordot/MatMulMatMul$dense_423/Tensordot/Reshape:output:0*dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_423/Tensordot/MatMul?
dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_423/Tensordot/Const_2?
!dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/concat_1/axis?
dense_423/Tensordot/concat_1ConcatV2%dense_423/Tensordot/GatherV2:output:0$dense_423/Tensordot/Const_2:output:0*dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat_1?
dense_423/TensordotReshape$dense_423/Tensordot/MatMul:product:0%dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot?
 dense_423/BiasAdd/ReadVariableOpReadVariableOp/dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02"
 dense_423/BiasAdd/ReadVariableOp?
dense_423/BiasAddBiasAdddense_423/Tensordot:output:0(dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_423/BiasAdd?
IdentityIdentitydense_423/BiasAdd:output:0!^dense_421/BiasAdd/ReadVariableOp#^dense_421/Tensordot/ReadVariableOp!^dense_422/BiasAdd/ReadVariableOp#^dense_422/Tensordot/ReadVariableOp!^dense_423/BiasAdd/ReadVariableOp#^dense_423/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_421/BiasAdd/ReadVariableOp dense_421/BiasAdd/ReadVariableOp2H
"dense_421/Tensordot/ReadVariableOp"dense_421/Tensordot/ReadVariableOp2D
 dense_422/BiasAdd/ReadVariableOp dense_422/BiasAdd/ReadVariableOp2H
"dense_422/Tensordot/ReadVariableOp"dense_422/Tensordot/ReadVariableOp2D
 dense_423/BiasAdd/ReadVariableOp dense_423/BiasAdd/ReadVariableOp2H
"dense_423/Tensordot/ReadVariableOp"dense_423/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_61001048

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
K__inference_discriminator_layer_call_and_return_conditional_losses_61000754

inputs7
3dense_424_tensordot_readvariableop_dense_424_kernel3
/dense_424_biasadd_readvariableop_dense_424_bias7
3dense_425_tensordot_readvariableop_dense_425_kernel3
/dense_425_biasadd_readvariableop_dense_425_bias7
3dense_426_tensordot_readvariableop_dense_426_kernel3
/dense_426_biasadd_readvariableop_dense_426_bias7
3dense_427_tensordot_readvariableop_dense_427_kernel3
/dense_427_biasadd_readvariableop_dense_427_bias4
0dense_428_matmul_readvariableop_dense_428_kernel3
/dense_428_biasadd_readvariableop_dense_428_bias
identity?? dense_424/BiasAdd/ReadVariableOp?"dense_424/Tensordot/ReadVariableOp? dense_425/BiasAdd/ReadVariableOp?"dense_425/Tensordot/ReadVariableOp? dense_426/BiasAdd/ReadVariableOp?"dense_426/Tensordot/ReadVariableOp? dense_427/BiasAdd/ReadVariableOp?"dense_427/Tensordot/ReadVariableOp? dense_428/BiasAdd/ReadVariableOp?dense_428/MatMul/ReadVariableOpw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulinputs dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_1/dropout/Mulh
dropout_1/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_1/dropout/Mul_1?
"dense_424/Tensordot/ReadVariableOpReadVariableOp3dense_424_tensordot_readvariableop_dense_424_kernel* 
_output_shapes
:
??*
dtype02$
"dense_424/Tensordot/ReadVariableOp~
dense_424/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_424/Tensordot/axes?
dense_424/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_424/Tensordot/free?
dense_424/Tensordot/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_424/Tensordot/Shape?
!dense_424/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_424/Tensordot/GatherV2/axis?
dense_424/Tensordot/GatherV2GatherV2"dense_424/Tensordot/Shape:output:0!dense_424/Tensordot/free:output:0*dense_424/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_424/Tensordot/GatherV2?
#dense_424/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_424/Tensordot/GatherV2_1/axis?
dense_424/Tensordot/GatherV2_1GatherV2"dense_424/Tensordot/Shape:output:0!dense_424/Tensordot/axes:output:0,dense_424/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_424/Tensordot/GatherV2_1?
dense_424/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_424/Tensordot/Const?
dense_424/Tensordot/ProdProd%dense_424/Tensordot/GatherV2:output:0"dense_424/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_424/Tensordot/Prod?
dense_424/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_424/Tensordot/Const_1?
dense_424/Tensordot/Prod_1Prod'dense_424/Tensordot/GatherV2_1:output:0$dense_424/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_424/Tensordot/Prod_1?
dense_424/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_424/Tensordot/concat/axis?
dense_424/Tensordot/concatConcatV2!dense_424/Tensordot/free:output:0!dense_424/Tensordot/axes:output:0(dense_424/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_424/Tensordot/concat?
dense_424/Tensordot/stackPack!dense_424/Tensordot/Prod:output:0#dense_424/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_424/Tensordot/stack?
dense_424/Tensordot/transpose	Transposedropout_1/dropout/Mul_1:z:0#dense_424/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_424/Tensordot/transpose?
dense_424/Tensordot/ReshapeReshape!dense_424/Tensordot/transpose:y:0"dense_424/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_424/Tensordot/Reshape?
dense_424/Tensordot/MatMulMatMul$dense_424/Tensordot/Reshape:output:0*dense_424/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_424/Tensordot/MatMul?
dense_424/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_424/Tensordot/Const_2?
!dense_424/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_424/Tensordot/concat_1/axis?
dense_424/Tensordot/concat_1ConcatV2%dense_424/Tensordot/GatherV2:output:0$dense_424/Tensordot/Const_2:output:0*dense_424/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_424/Tensordot/concat_1?
dense_424/TensordotReshape$dense_424/Tensordot/MatMul:product:0%dense_424/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_424/Tensordot?
 dense_424/BiasAdd/ReadVariableOpReadVariableOp/dense_424_biasadd_readvariableop_dense_424_bias*
_output_shapes	
:?*
dtype02"
 dense_424/BiasAdd/ReadVariableOp?
dense_424/BiasAddBiasAdddense_424/Tensordot:output:0(dense_424/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_424/BiasAdd{
dense_424/TanhTanhdense_424/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_424/Tanh?
"dense_425/Tensordot/ReadVariableOpReadVariableOp3dense_425_tensordot_readvariableop_dense_425_kernel* 
_output_shapes
:
??*
dtype02$
"dense_425/Tensordot/ReadVariableOp~
dense_425/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_425/Tensordot/axes?
dense_425/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_425/Tensordot/freex
dense_425/Tensordot/ShapeShapedense_424/Tanh:y:0*
T0*
_output_shapes
:2
dense_425/Tensordot/Shape?
!dense_425/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_425/Tensordot/GatherV2/axis?
dense_425/Tensordot/GatherV2GatherV2"dense_425/Tensordot/Shape:output:0!dense_425/Tensordot/free:output:0*dense_425/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_425/Tensordot/GatherV2?
#dense_425/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_425/Tensordot/GatherV2_1/axis?
dense_425/Tensordot/GatherV2_1GatherV2"dense_425/Tensordot/Shape:output:0!dense_425/Tensordot/axes:output:0,dense_425/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_425/Tensordot/GatherV2_1?
dense_425/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_425/Tensordot/Const?
dense_425/Tensordot/ProdProd%dense_425/Tensordot/GatherV2:output:0"dense_425/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_425/Tensordot/Prod?
dense_425/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_425/Tensordot/Const_1?
dense_425/Tensordot/Prod_1Prod'dense_425/Tensordot/GatherV2_1:output:0$dense_425/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_425/Tensordot/Prod_1?
dense_425/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_425/Tensordot/concat/axis?
dense_425/Tensordot/concatConcatV2!dense_425/Tensordot/free:output:0!dense_425/Tensordot/axes:output:0(dense_425/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_425/Tensordot/concat?
dense_425/Tensordot/stackPack!dense_425/Tensordot/Prod:output:0#dense_425/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_425/Tensordot/stack?
dense_425/Tensordot/transpose	Transposedense_424/Tanh:y:0#dense_425/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_425/Tensordot/transpose?
dense_425/Tensordot/ReshapeReshape!dense_425/Tensordot/transpose:y:0"dense_425/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_425/Tensordot/Reshape?
dense_425/Tensordot/MatMulMatMul$dense_425/Tensordot/Reshape:output:0*dense_425/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_425/Tensordot/MatMul?
dense_425/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_425/Tensordot/Const_2?
!dense_425/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_425/Tensordot/concat_1/axis?
dense_425/Tensordot/concat_1ConcatV2%dense_425/Tensordot/GatherV2:output:0$dense_425/Tensordot/Const_2:output:0*dense_425/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_425/Tensordot/concat_1?
dense_425/TensordotReshape$dense_425/Tensordot/MatMul:product:0%dense_425/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_425/Tensordot?
 dense_425/BiasAdd/ReadVariableOpReadVariableOp/dense_425_biasadd_readvariableop_dense_425_bias*
_output_shapes	
:?*
dtype02"
 dense_425/BiasAdd/ReadVariableOp?
dense_425/BiasAddBiasAdddense_425/Tensordot:output:0(dense_425/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_425/BiasAdd{
dense_425/TanhTanhdense_425/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_425/Tanh?
"dense_426/Tensordot/ReadVariableOpReadVariableOp3dense_426_tensordot_readvariableop_dense_426_kernel* 
_output_shapes
:
??*
dtype02$
"dense_426/Tensordot/ReadVariableOp~
dense_426/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_426/Tensordot/axes?
dense_426/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_426/Tensordot/freex
dense_426/Tensordot/ShapeShapedense_425/Tanh:y:0*
T0*
_output_shapes
:2
dense_426/Tensordot/Shape?
!dense_426/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_426/Tensordot/GatherV2/axis?
dense_426/Tensordot/GatherV2GatherV2"dense_426/Tensordot/Shape:output:0!dense_426/Tensordot/free:output:0*dense_426/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_426/Tensordot/GatherV2?
#dense_426/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_426/Tensordot/GatherV2_1/axis?
dense_426/Tensordot/GatherV2_1GatherV2"dense_426/Tensordot/Shape:output:0!dense_426/Tensordot/axes:output:0,dense_426/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_426/Tensordot/GatherV2_1?
dense_426/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_426/Tensordot/Const?
dense_426/Tensordot/ProdProd%dense_426/Tensordot/GatherV2:output:0"dense_426/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_426/Tensordot/Prod?
dense_426/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_426/Tensordot/Const_1?
dense_426/Tensordot/Prod_1Prod'dense_426/Tensordot/GatherV2_1:output:0$dense_426/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_426/Tensordot/Prod_1?
dense_426/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_426/Tensordot/concat/axis?
dense_426/Tensordot/concatConcatV2!dense_426/Tensordot/free:output:0!dense_426/Tensordot/axes:output:0(dense_426/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_426/Tensordot/concat?
dense_426/Tensordot/stackPack!dense_426/Tensordot/Prod:output:0#dense_426/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_426/Tensordot/stack?
dense_426/Tensordot/transpose	Transposedense_425/Tanh:y:0#dense_426/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_426/Tensordot/transpose?
dense_426/Tensordot/ReshapeReshape!dense_426/Tensordot/transpose:y:0"dense_426/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_426/Tensordot/Reshape?
dense_426/Tensordot/MatMulMatMul$dense_426/Tensordot/Reshape:output:0*dense_426/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_426/Tensordot/MatMul?
dense_426/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_426/Tensordot/Const_2?
!dense_426/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_426/Tensordot/concat_1/axis?
dense_426/Tensordot/concat_1ConcatV2%dense_426/Tensordot/GatherV2:output:0$dense_426/Tensordot/Const_2:output:0*dense_426/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_426/Tensordot/concat_1?
dense_426/TensordotReshape$dense_426/Tensordot/MatMul:product:0%dense_426/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_426/Tensordot?
 dense_426/BiasAdd/ReadVariableOpReadVariableOp/dense_426_biasadd_readvariableop_dense_426_bias*
_output_shapes	
:?*
dtype02"
 dense_426/BiasAdd/ReadVariableOp?
dense_426/BiasAddBiasAdddense_426/Tensordot:output:0(dense_426/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_426/BiasAdd{
dense_426/TanhTanhdense_426/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_426/Tanh?
"dense_427/Tensordot/ReadVariableOpReadVariableOp3dense_427_tensordot_readvariableop_dense_427_kernel*
_output_shapes
:	?*
dtype02$
"dense_427/Tensordot/ReadVariableOp~
dense_427/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_427/Tensordot/axes?
dense_427/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_427/Tensordot/freex
dense_427/Tensordot/ShapeShapedense_426/Tanh:y:0*
T0*
_output_shapes
:2
dense_427/Tensordot/Shape?
!dense_427/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_427/Tensordot/GatherV2/axis?
dense_427/Tensordot/GatherV2GatherV2"dense_427/Tensordot/Shape:output:0!dense_427/Tensordot/free:output:0*dense_427/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_427/Tensordot/GatherV2?
#dense_427/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_427/Tensordot/GatherV2_1/axis?
dense_427/Tensordot/GatherV2_1GatherV2"dense_427/Tensordot/Shape:output:0!dense_427/Tensordot/axes:output:0,dense_427/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_427/Tensordot/GatherV2_1?
dense_427/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_427/Tensordot/Const?
dense_427/Tensordot/ProdProd%dense_427/Tensordot/GatherV2:output:0"dense_427/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_427/Tensordot/Prod?
dense_427/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_427/Tensordot/Const_1?
dense_427/Tensordot/Prod_1Prod'dense_427/Tensordot/GatherV2_1:output:0$dense_427/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_427/Tensordot/Prod_1?
dense_427/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_427/Tensordot/concat/axis?
dense_427/Tensordot/concatConcatV2!dense_427/Tensordot/free:output:0!dense_427/Tensordot/axes:output:0(dense_427/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_427/Tensordot/concat?
dense_427/Tensordot/stackPack!dense_427/Tensordot/Prod:output:0#dense_427/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_427/Tensordot/stack?
dense_427/Tensordot/transpose	Transposedense_426/Tanh:y:0#dense_427/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_427/Tensordot/transpose?
dense_427/Tensordot/ReshapeReshape!dense_427/Tensordot/transpose:y:0"dense_427/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_427/Tensordot/Reshape?
dense_427/Tensordot/MatMulMatMul$dense_427/Tensordot/Reshape:output:0*dense_427/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_427/Tensordot/MatMul?
dense_427/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_427/Tensordot/Const_2?
!dense_427/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_427/Tensordot/concat_1/axis?
dense_427/Tensordot/concat_1ConcatV2%dense_427/Tensordot/GatherV2:output:0$dense_427/Tensordot/Const_2:output:0*dense_427/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_427/Tensordot/concat_1?
dense_427/TensordotReshape$dense_427/Tensordot/MatMul:product:0%dense_427/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_427/Tensordot?
 dense_427/BiasAdd/ReadVariableOpReadVariableOp/dense_427_biasadd_readvariableop_dense_427_bias*
_output_shapes
:*
dtype02"
 dense_427/BiasAdd/ReadVariableOp?
dense_427/BiasAddBiasAdddense_427/Tensordot:output:0(dense_427/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_427/BiasAddz
dense_427/TanhTanhdense_427/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense_427/Tanhu
flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_52/Const?
flatten_52/ReshapeReshapedense_427/Tanh:y:0flatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_52/Reshape?
dense_428/MatMul/ReadVariableOpReadVariableOp0dense_428_matmul_readvariableop_dense_428_kernel*
_output_shapes

:*
dtype02!
dense_428/MatMul/ReadVariableOp?
dense_428/MatMulMatMulflatten_52/Reshape:output:0'dense_428/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_428/MatMul?
 dense_428/BiasAdd/ReadVariableOpReadVariableOp/dense_428_biasadd_readvariableop_dense_428_bias*
_output_shapes
:*
dtype02"
 dense_428/BiasAdd/ReadVariableOp?
dense_428/BiasAddBiasAdddense_428/MatMul:product:0(dense_428/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_428/BiasAdd
dense_428/SigmoidSigmoiddense_428/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_428/Sigmoid?
IdentityIdentitydense_428/Sigmoid:y:0!^dense_424/BiasAdd/ReadVariableOp#^dense_424/Tensordot/ReadVariableOp!^dense_425/BiasAdd/ReadVariableOp#^dense_425/Tensordot/ReadVariableOp!^dense_426/BiasAdd/ReadVariableOp#^dense_426/Tensordot/ReadVariableOp!^dense_427/BiasAdd/ReadVariableOp#^dense_427/Tensordot/ReadVariableOp!^dense_428/BiasAdd/ReadVariableOp ^dense_428/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::2D
 dense_424/BiasAdd/ReadVariableOp dense_424/BiasAdd/ReadVariableOp2H
"dense_424/Tensordot/ReadVariableOp"dense_424/Tensordot/ReadVariableOp2D
 dense_425/BiasAdd/ReadVariableOp dense_425/BiasAdd/ReadVariableOp2H
"dense_425/Tensordot/ReadVariableOp"dense_425/Tensordot/ReadVariableOp2D
 dense_426/BiasAdd/ReadVariableOp dense_426/BiasAdd/ReadVariableOp2H
"dense_426/Tensordot/ReadVariableOp"dense_426/Tensordot/ReadVariableOp2D
 dense_427/BiasAdd/ReadVariableOp dense_427/BiasAdd/ReadVariableOp2H
"dense_427/Tensordot/ReadVariableOp"dense_427/Tensordot/ReadVariableOp2D
 dense_428/BiasAdd/ReadVariableOp dense_428/BiasAdd/ReadVariableOp2B
dense_428/MatMul/ReadVariableOpdense_428/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?q
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000603
inputs_07
3dense_421_tensordot_readvariableop_dense_421_kernel3
/dense_421_biasadd_readvariableop_dense_421_bias7
3dense_422_tensordot_readvariableop_dense_422_kernel3
/dense_422_biasadd_readvariableop_dense_422_bias7
3dense_423_tensordot_readvariableop_dense_423_kernel3
/dense_423_biasadd_readvariableop_dense_423_bias
identity?? dense_421/BiasAdd/ReadVariableOp?"dense_421/Tensordot/ReadVariableOp? dense_422/BiasAdd/ReadVariableOp?"dense_422/Tensordot/ReadVariableOp? dense_423/BiasAdd/ReadVariableOp?"dense_423/Tensordot/ReadVariableOp?
"dense_421/Tensordot/ReadVariableOpReadVariableOp3dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02$
"dense_421/Tensordot/ReadVariableOp~
dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_421/Tensordot/axes?
dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_421/Tensordot/freen
dense_421/Tensordot/ShapeShapeinputs_0*
T0*
_output_shapes
:2
dense_421/Tensordot/Shape?
!dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/GatherV2/axis?
dense_421/Tensordot/GatherV2GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/free:output:0*dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_421/Tensordot/GatherV2?
#dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_421/Tensordot/GatherV2_1/axis?
dense_421/Tensordot/GatherV2_1GatherV2"dense_421/Tensordot/Shape:output:0!dense_421/Tensordot/axes:output:0,dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_421/Tensordot/GatherV2_1?
dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const?
dense_421/Tensordot/ProdProd%dense_421/Tensordot/GatherV2:output:0"dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod?
dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_1?
dense_421/Tensordot/Prod_1Prod'dense_421/Tensordot/GatherV2_1:output:0$dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_421/Tensordot/Prod_1?
dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_421/Tensordot/concat/axis?
dense_421/Tensordot/concatConcatV2!dense_421/Tensordot/free:output:0!dense_421/Tensordot/axes:output:0(dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat?
dense_421/Tensordot/stackPack!dense_421/Tensordot/Prod:output:0#dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/stack?
dense_421/Tensordot/transpose	Transposeinputs_0#dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_421/Tensordot/transpose?
dense_421/Tensordot/ReshapeReshape!dense_421/Tensordot/transpose:y:0"dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_421/Tensordot/Reshape?
dense_421/Tensordot/MatMulMatMul$dense_421/Tensordot/Reshape:output:0*dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_421/Tensordot/MatMul?
dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_421/Tensordot/Const_2?
!dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_421/Tensordot/concat_1/axis?
dense_421/Tensordot/concat_1ConcatV2%dense_421/Tensordot/GatherV2:output:0$dense_421/Tensordot/Const_2:output:0*dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_421/Tensordot/concat_1?
dense_421/TensordotReshape$dense_421/Tensordot/MatMul:product:0%dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tensordot?
 dense_421/BiasAdd/ReadVariableOpReadVariableOp/dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02"
 dense_421/BiasAdd/ReadVariableOp?
dense_421/BiasAddBiasAdddense_421/Tensordot:output:0(dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_421/BiasAddz
dense_421/TanhTanhdense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_421/Tanh?
"dense_422/Tensordot/ReadVariableOpReadVariableOp3dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_422/Tensordot/ReadVariableOp~
dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_422/Tensordot/axes?
dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_422/Tensordot/freex
dense_422/Tensordot/ShapeShapedense_421/Tanh:y:0*
T0*
_output_shapes
:2
dense_422/Tensordot/Shape?
!dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/GatherV2/axis?
dense_422/Tensordot/GatherV2GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/free:output:0*dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_422/Tensordot/GatherV2?
#dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_422/Tensordot/GatherV2_1/axis?
dense_422/Tensordot/GatherV2_1GatherV2"dense_422/Tensordot/Shape:output:0!dense_422/Tensordot/axes:output:0,dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_422/Tensordot/GatherV2_1?
dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const?
dense_422/Tensordot/ProdProd%dense_422/Tensordot/GatherV2:output:0"dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod?
dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_422/Tensordot/Const_1?
dense_422/Tensordot/Prod_1Prod'dense_422/Tensordot/GatherV2_1:output:0$dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_422/Tensordot/Prod_1?
dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_422/Tensordot/concat/axis?
dense_422/Tensordot/concatConcatV2!dense_422/Tensordot/free:output:0!dense_422/Tensordot/axes:output:0(dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat?
dense_422/Tensordot/stackPack!dense_422/Tensordot/Prod:output:0#dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/stack?
dense_422/Tensordot/transpose	Transposedense_421/Tanh:y:0#dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_422/Tensordot/transpose?
dense_422/Tensordot/ReshapeReshape!dense_422/Tensordot/transpose:y:0"dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_422/Tensordot/Reshape?
dense_422/Tensordot/MatMulMatMul$dense_422/Tensordot/Reshape:output:0*dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_422/Tensordot/MatMul?
dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_422/Tensordot/Const_2?
!dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_422/Tensordot/concat_1/axis?
dense_422/Tensordot/concat_1ConcatV2%dense_422/Tensordot/GatherV2:output:0$dense_422/Tensordot/Const_2:output:0*dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_422/Tensordot/concat_1?
dense_422/TensordotReshape$dense_422/Tensordot/MatMul:product:0%dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tensordot?
 dense_422/BiasAdd/ReadVariableOpReadVariableOp/dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02"
 dense_422/BiasAdd/ReadVariableOp?
dense_422/BiasAddBiasAdddense_422/Tensordot:output:0(dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_422/BiasAdd{
dense_422/TanhTanhdense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_422/Tanhv

add_53/addAddV2inputs_0dense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2

add_53/add?
"dense_423/Tensordot/ReadVariableOpReadVariableOp3dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02$
"dense_423/Tensordot/ReadVariableOp~
dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_423/Tensordot/axes?
dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_423/Tensordot/freet
dense_423/Tensordot/ShapeShapeadd_53/add:z:0*
T0*
_output_shapes
:2
dense_423/Tensordot/Shape?
!dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/GatherV2/axis?
dense_423/Tensordot/GatherV2GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/free:output:0*dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_423/Tensordot/GatherV2?
#dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_423/Tensordot/GatherV2_1/axis?
dense_423/Tensordot/GatherV2_1GatherV2"dense_423/Tensordot/Shape:output:0!dense_423/Tensordot/axes:output:0,dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_423/Tensordot/GatherV2_1?
dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const?
dense_423/Tensordot/ProdProd%dense_423/Tensordot/GatherV2:output:0"dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod?
dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_423/Tensordot/Const_1?
dense_423/Tensordot/Prod_1Prod'dense_423/Tensordot/GatherV2_1:output:0$dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_423/Tensordot/Prod_1?
dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_423/Tensordot/concat/axis?
dense_423/Tensordot/concatConcatV2!dense_423/Tensordot/free:output:0!dense_423/Tensordot/axes:output:0(dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat?
dense_423/Tensordot/stackPack!dense_423/Tensordot/Prod:output:0#dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/stack?
dense_423/Tensordot/transpose	Transposeadd_53/add:z:0#dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot/transpose?
dense_423/Tensordot/ReshapeReshape!dense_423/Tensordot/transpose:y:0"dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_423/Tensordot/Reshape?
dense_423/Tensordot/MatMulMatMul$dense_423/Tensordot/Reshape:output:0*dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_423/Tensordot/MatMul?
dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_423/Tensordot/Const_2?
!dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_423/Tensordot/concat_1/axis?
dense_423/Tensordot/concat_1ConcatV2%dense_423/Tensordot/GatherV2:output:0$dense_423/Tensordot/Const_2:output:0*dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_423/Tensordot/concat_1?
dense_423/TensordotReshape$dense_423/Tensordot/MatMul:product:0%dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_423/Tensordot?
 dense_423/BiasAdd/ReadVariableOpReadVariableOp/dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02"
 dense_423/BiasAdd/ReadVariableOp?
dense_423/BiasAddBiasAdddense_423/Tensordot:output:0(dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_423/BiasAdd?
IdentityIdentitydense_423/BiasAdd:output:0!^dense_421/BiasAdd/ReadVariableOp#^dense_421/Tensordot/ReadVariableOp!^dense_422/BiasAdd/ReadVariableOp#^dense_422/Tensordot/ReadVariableOp!^dense_423/BiasAdd/ReadVariableOp#^dense_423/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_421/BiasAdd/ReadVariableOp dense_421/BiasAdd/ReadVariableOp2H
"dense_421/Tensordot/ReadVariableOp"dense_421/Tensordot/ReadVariableOp2D
 dense_422/BiasAdd/ReadVariableOp dense_422/BiasAdd/ReadVariableOp2H
"dense_422/Tensordot/ReadVariableOp"dense_422/Tensordot/ReadVariableOp2D
 dense_423/BiasAdd/ReadVariableOp dense_423/BiasAdd/ReadVariableOp2H
"dense_423/Tensordot/ReadVariableOp"dense_423/Tensordot/ReadVariableOp:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0
? 
?
G__inference_dense_427_layer_call_and_return_conditional_losses_61001203

inputs-
)tensordot_readvariableop_dense_427_kernel)
%biasadd_readvariableop_dense_427_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_427_kernel*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_427_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_52_layer_call_and_return_conditional_losses_61001216

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_model_975_layer_call_fn_61000220

inputs
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_biasdense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_975_layer_call_and_return_conditional_losses_609997012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_421_layer_call_and_return_conditional_losses_61000937

inputs-
)tensordot_readvariableop_dense_421_kernel)
%biasadd_readvariableop_dense_421_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
U
)__inference_add_53_layer_call_fn_61000994
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_53_layer_call_and_return_conditional_losses_609989542
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????:??????????:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999062

inputs
dense_421_dense_421_kernel
dense_421_dense_421_bias
dense_422_dense_422_kernel
dense_422_dense_422_bias
dense_423_dense_423_kernel
dense_423_dense_423_bias
identity??!dense_421/StatefulPartitionedCall?!dense_422/StatefulPartitionedCall?!dense_423/StatefulPartitionedCall?
!dense_421/StatefulPartitionedCallStatefulPartitionedCallinputsdense_421_dense_421_kerneldense_421_dense_421_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_421_layer_call_and_return_conditional_losses_609988932#
!dense_421/StatefulPartitionedCall?
!dense_422/StatefulPartitionedCallStatefulPartitionedCall*dense_421/StatefulPartitionedCall:output:0dense_422_dense_422_kerneldense_422_dense_422_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_422_layer_call_and_return_conditional_losses_609989362#
!dense_422/StatefulPartitionedCall?
add_53/PartitionedCallPartitionedCallinputs*dense_422/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_53_layer_call_and_return_conditional_losses_609989542
add_53/PartitionedCall?
!dense_423/StatefulPartitionedCallStatefulPartitionedCalladd_53/PartitionedCall:output:0dense_423_dense_423_kerneldense_423_dense_423_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_423_layer_call_and_return_conditional_losses_609989932#
!dense_423/StatefulPartitionedCall?
IdentityIdentity*dense_423/StatefulPartitionedCall:output:0"^dense_421/StatefulPartitionedCall"^dense_422/StatefulPartitionedCall"^dense_423/StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2F
!dense_421/StatefulPartitionedCall!dense_421/StatefulPartitionedCall2F
!dense_422/StatefulPartitionedCall!dense_422/StatefulPartitionedCall2F
!dense_423/StatefulPartitionedCall!dense_423/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
p
D__inference_add_53_layer_call_and_return_conditional_losses_61000988
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:??????????2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????:??????????:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
n
D__inference_add_53_layer_call_and_return_conditional_losses_60998954

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:??????????2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:TP
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999006
input_13
dense_421_dense_421_kernel
dense_421_dense_421_bias
dense_422_dense_422_kernel
dense_422_dense_422_bias
dense_423_dense_423_kernel
dense_423_dense_423_bias
identity??!dense_421/StatefulPartitionedCall?!dense_422/StatefulPartitionedCall?!dense_423/StatefulPartitionedCall?
!dense_421/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_421_dense_421_kerneldense_421_dense_421_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_421_layer_call_and_return_conditional_losses_609988932#
!dense_421/StatefulPartitionedCall?
!dense_422/StatefulPartitionedCallStatefulPartitionedCall*dense_421/StatefulPartitionedCall:output:0dense_422_dense_422_kerneldense_422_dense_422_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_422_layer_call_and_return_conditional_losses_609989362#
!dense_422/StatefulPartitionedCall?
add_53/PartitionedCallPartitionedCallinput_13*dense_422/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_add_53_layer_call_and_return_conditional_losses_609989542
add_53/PartitionedCall?
!dense_423/StatefulPartitionedCallStatefulPartitionedCalladd_53/PartitionedCall:output:0dense_423_dense_423_kerneldense_423_dense_423_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_423_layer_call_and_return_conditional_losses_609989932#
!dense_423/StatefulPartitionedCall?
IdentityIdentity*dense_423/StatefulPartitionedCall:output:0"^dense_421/StatefulPartitionedCall"^dense_422/StatefulPartitionedCall"^dense_423/StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2F
!dense_421/StatefulPartitionedCall!dense_421/StatefulPartitionedCall2F
!dense_422/StatefulPartitionedCall!dense_422/StatefulPartitionedCall2F
!dense_423/StatefulPartitionedCall!dense_423/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
?
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_61001043

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
K__inference_discriminator_layer_call_and_return_conditional_losses_60999360

inputs
dense_424_dense_424_kernel
dense_424_dense_424_bias
dense_425_dense_425_kernel
dense_425_dense_425_bias
dense_426_dense_426_kernel
dense_426_dense_426_bias
dense_427_dense_427_kernel
dense_427_dense_427_bias
dense_428_dense_428_kernel
dense_428_dense_428_bias
identity??!dense_424/StatefulPartitionedCall?!dense_425/StatefulPartitionedCall?!dense_426/StatefulPartitionedCall?!dense_427/StatefulPartitionedCall?!dense_428/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_609990872#
!dropout_1/StatefulPartitionedCall?
!dense_424/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_424_dense_424_kerneldense_424_dense_424_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_424_layer_call_and_return_conditional_losses_609991362#
!dense_424/StatefulPartitionedCall?
!dense_425/StatefulPartitionedCallStatefulPartitionedCall*dense_424/StatefulPartitionedCall:output:0dense_425_dense_425_kerneldense_425_dense_425_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_425_layer_call_and_return_conditional_losses_609991792#
!dense_425/StatefulPartitionedCall?
!dense_426/StatefulPartitionedCallStatefulPartitionedCall*dense_425/StatefulPartitionedCall:output:0dense_426_dense_426_kerneldense_426_dense_426_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_426_layer_call_and_return_conditional_losses_609992222#
!dense_426/StatefulPartitionedCall?
!dense_427/StatefulPartitionedCallStatefulPartitionedCall*dense_426/StatefulPartitionedCall:output:0dense_427_dense_427_kerneldense_427_dense_427_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_427_layer_call_and_return_conditional_losses_609992652#
!dense_427/StatefulPartitionedCall?
flatten_52/PartitionedCallPartitionedCall*dense_427/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_52_layer_call_and_return_conditional_losses_609992832
flatten_52/PartitionedCall?
!dense_428/StatefulPartitionedCallStatefulPartitionedCall#flatten_52/PartitionedCall:output:0dense_428_dense_428_kerneldense_428_dense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_428_layer_call_and_return_conditional_losses_609993022#
!dense_428/StatefulPartitionedCall?
IdentityIdentity*dense_428/StatefulPartitionedCall:output:0"^dense_424/StatefulPartitionedCall"^dense_425/StatefulPartitionedCall"^dense_426/StatefulPartitionedCall"^dense_427/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::2F
!dense_424/StatefulPartitionedCall!dense_424/StatefulPartitionedCall2F
!dense_425/StatefulPartitionedCall!dense_425/StatefulPartitionedCall2F
!dense_426/StatefulPartitionedCall!dense_426/StatefulPartitionedCall2F
!dense_427/StatefulPartitionedCall!dense_427/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_428_layer_call_fn_61001239

inputs
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_428_layer_call_and_return_conditional_losses_609993022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_427_layer_call_and_return_conditional_losses_60999265

inputs-
)tensordot_readvariableop_dense_427_kernel)
%biasadd_readvariableop_dense_427_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_427_kernel*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_427_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
0__inference_discriminator_layer_call_fn_60999409
input_14
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14dense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609993962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_14
? 
?
G__inference_dense_422_layer_call_and_return_conditional_losses_60998936

inputs-
)tensordot_readvariableop_dense_422_kernel)
%biasadd_readvariableop_dense_422_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
H
,__inference_dropout_1_layer_call_fn_61001058

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_609990922
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
G__inference_model_975_layer_call_and_return_conditional_losses_61000199

inputsC
?autoencoder_dense_421_tensordot_readvariableop_dense_421_kernel?
;autoencoder_dense_421_biasadd_readvariableop_dense_421_biasC
?autoencoder_dense_422_tensordot_readvariableop_dense_422_kernel?
;autoencoder_dense_422_biasadd_readvariableop_dense_422_biasC
?autoencoder_dense_423_tensordot_readvariableop_dense_423_kernel?
;autoencoder_dense_423_biasadd_readvariableop_dense_423_biasE
Adiscriminator_dense_424_tensordot_readvariableop_dense_424_kernelA
=discriminator_dense_424_biasadd_readvariableop_dense_424_biasE
Adiscriminator_dense_425_tensordot_readvariableop_dense_425_kernelA
=discriminator_dense_425_biasadd_readvariableop_dense_425_biasE
Adiscriminator_dense_426_tensordot_readvariableop_dense_426_kernelA
=discriminator_dense_426_biasadd_readvariableop_dense_426_biasE
Adiscriminator_dense_427_tensordot_readvariableop_dense_427_kernelA
=discriminator_dense_427_biasadd_readvariableop_dense_427_biasB
>discriminator_dense_428_matmul_readvariableop_dense_428_kernelA
=discriminator_dense_428_biasadd_readvariableop_dense_428_bias
identity??,autoencoder/dense_421/BiasAdd/ReadVariableOp?.autoencoder/dense_421/Tensordot/ReadVariableOp?,autoencoder/dense_422/BiasAdd/ReadVariableOp?.autoencoder/dense_422/Tensordot/ReadVariableOp?,autoencoder/dense_423/BiasAdd/ReadVariableOp?.autoencoder/dense_423/Tensordot/ReadVariableOp?.discriminator/dense_424/BiasAdd/ReadVariableOp?0discriminator/dense_424/Tensordot/ReadVariableOp?.discriminator/dense_425/BiasAdd/ReadVariableOp?0discriminator/dense_425/Tensordot/ReadVariableOp?.discriminator/dense_426/BiasAdd/ReadVariableOp?0discriminator/dense_426/Tensordot/ReadVariableOp?.discriminator/dense_427/BiasAdd/ReadVariableOp?0discriminator/dense_427/Tensordot/ReadVariableOp?.discriminator/dense_428/BiasAdd/ReadVariableOp?-discriminator/dense_428/MatMul/ReadVariableOp?
.autoencoder/dense_421/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_421_tensordot_readvariableop_dense_421_kernel*
_output_shapes
:	? *
dtype020
.autoencoder/dense_421/Tensordot/ReadVariableOp?
$autoencoder/dense_421/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_421/Tensordot/axes?
$autoencoder/dense_421/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_421/Tensordot/free?
%autoencoder/dense_421/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2'
%autoencoder/dense_421/Tensordot/Shape?
-autoencoder/dense_421/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_421/Tensordot/GatherV2/axis?
(autoencoder/dense_421/Tensordot/GatherV2GatherV2.autoencoder/dense_421/Tensordot/Shape:output:0-autoencoder/dense_421/Tensordot/free:output:06autoencoder/dense_421/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_421/Tensordot/GatherV2?
/autoencoder/dense_421/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_421/Tensordot/GatherV2_1/axis?
*autoencoder/dense_421/Tensordot/GatherV2_1GatherV2.autoencoder/dense_421/Tensordot/Shape:output:0-autoencoder/dense_421/Tensordot/axes:output:08autoencoder/dense_421/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_421/Tensordot/GatherV2_1?
%autoencoder/dense_421/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_421/Tensordot/Const?
$autoencoder/dense_421/Tensordot/ProdProd1autoencoder/dense_421/Tensordot/GatherV2:output:0.autoencoder/dense_421/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_421/Tensordot/Prod?
'autoencoder/dense_421/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_421/Tensordot/Const_1?
&autoencoder/dense_421/Tensordot/Prod_1Prod3autoencoder/dense_421/Tensordot/GatherV2_1:output:00autoencoder/dense_421/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_421/Tensordot/Prod_1?
+autoencoder/dense_421/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_421/Tensordot/concat/axis?
&autoencoder/dense_421/Tensordot/concatConcatV2-autoencoder/dense_421/Tensordot/free:output:0-autoencoder/dense_421/Tensordot/axes:output:04autoencoder/dense_421/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_421/Tensordot/concat?
%autoencoder/dense_421/Tensordot/stackPack-autoencoder/dense_421/Tensordot/Prod:output:0/autoencoder/dense_421/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_421/Tensordot/stack?
)autoencoder/dense_421/Tensordot/transpose	Transposeinputs/autoencoder/dense_421/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)autoencoder/dense_421/Tensordot/transpose?
'autoencoder/dense_421/Tensordot/ReshapeReshape-autoencoder/dense_421/Tensordot/transpose:y:0.autoencoder/dense_421/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_421/Tensordot/Reshape?
&autoencoder/dense_421/Tensordot/MatMulMatMul0autoencoder/dense_421/Tensordot/Reshape:output:06autoencoder/dense_421/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&autoencoder/dense_421/Tensordot/MatMul?
'autoencoder/dense_421/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_421/Tensordot/Const_2?
-autoencoder/dense_421/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_421/Tensordot/concat_1/axis?
(autoencoder/dense_421/Tensordot/concat_1ConcatV21autoencoder/dense_421/Tensordot/GatherV2:output:00autoencoder/dense_421/Tensordot/Const_2:output:06autoencoder/dense_421/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_421/Tensordot/concat_1?
autoencoder/dense_421/TensordotReshape0autoencoder/dense_421/Tensordot/MatMul:product:01autoencoder/dense_421/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2!
autoencoder/dense_421/Tensordot?
,autoencoder/dense_421/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_421_biasadd_readvariableop_dense_421_bias*
_output_shapes
: *
dtype02.
,autoencoder/dense_421/BiasAdd/ReadVariableOp?
autoencoder/dense_421/BiasAddBiasAdd(autoencoder/dense_421/Tensordot:output:04autoencoder/dense_421/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
autoencoder/dense_421/BiasAdd?
autoencoder/dense_421/TanhTanh&autoencoder/dense_421/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
autoencoder/dense_421/Tanh?
.autoencoder/dense_422/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_422_tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype020
.autoencoder/dense_422/Tensordot/ReadVariableOp?
$autoencoder/dense_422/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_422/Tensordot/axes?
$autoencoder/dense_422/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_422/Tensordot/free?
%autoencoder/dense_422/Tensordot/ShapeShapeautoencoder/dense_421/Tanh:y:0*
T0*
_output_shapes
:2'
%autoencoder/dense_422/Tensordot/Shape?
-autoencoder/dense_422/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_422/Tensordot/GatherV2/axis?
(autoencoder/dense_422/Tensordot/GatherV2GatherV2.autoencoder/dense_422/Tensordot/Shape:output:0-autoencoder/dense_422/Tensordot/free:output:06autoencoder/dense_422/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_422/Tensordot/GatherV2?
/autoencoder/dense_422/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_422/Tensordot/GatherV2_1/axis?
*autoencoder/dense_422/Tensordot/GatherV2_1GatherV2.autoencoder/dense_422/Tensordot/Shape:output:0-autoencoder/dense_422/Tensordot/axes:output:08autoencoder/dense_422/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_422/Tensordot/GatherV2_1?
%autoencoder/dense_422/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_422/Tensordot/Const?
$autoencoder/dense_422/Tensordot/ProdProd1autoencoder/dense_422/Tensordot/GatherV2:output:0.autoencoder/dense_422/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_422/Tensordot/Prod?
'autoencoder/dense_422/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_422/Tensordot/Const_1?
&autoencoder/dense_422/Tensordot/Prod_1Prod3autoencoder/dense_422/Tensordot/GatherV2_1:output:00autoencoder/dense_422/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_422/Tensordot/Prod_1?
+autoencoder/dense_422/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_422/Tensordot/concat/axis?
&autoencoder/dense_422/Tensordot/concatConcatV2-autoencoder/dense_422/Tensordot/free:output:0-autoencoder/dense_422/Tensordot/axes:output:04autoencoder/dense_422/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_422/Tensordot/concat?
%autoencoder/dense_422/Tensordot/stackPack-autoencoder/dense_422/Tensordot/Prod:output:0/autoencoder/dense_422/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_422/Tensordot/stack?
)autoencoder/dense_422/Tensordot/transpose	Transposeautoencoder/dense_421/Tanh:y:0/autoencoder/dense_422/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2+
)autoencoder/dense_422/Tensordot/transpose?
'autoencoder/dense_422/Tensordot/ReshapeReshape-autoencoder/dense_422/Tensordot/transpose:y:0.autoencoder/dense_422/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_422/Tensordot/Reshape?
&autoencoder/dense_422/Tensordot/MatMulMatMul0autoencoder/dense_422/Tensordot/Reshape:output:06autoencoder/dense_422/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/dense_422/Tensordot/MatMul?
'autoencoder/dense_422/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'autoencoder/dense_422/Tensordot/Const_2?
-autoencoder/dense_422/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_422/Tensordot/concat_1/axis?
(autoencoder/dense_422/Tensordot/concat_1ConcatV21autoencoder/dense_422/Tensordot/GatherV2:output:00autoencoder/dense_422/Tensordot/Const_2:output:06autoencoder/dense_422/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_422/Tensordot/concat_1?
autoencoder/dense_422/TensordotReshape0autoencoder/dense_422/Tensordot/MatMul:product:01autoencoder/dense_422/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
autoencoder/dense_422/Tensordot?
,autoencoder/dense_422/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_422_biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02.
,autoencoder/dense_422/BiasAdd/ReadVariableOp?
autoencoder/dense_422/BiasAddBiasAdd(autoencoder/dense_422/Tensordot:output:04autoencoder/dense_422/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_422/BiasAdd?
autoencoder/dense_422/TanhTanh&autoencoder/dense_422/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_422/Tanh?
autoencoder/add_53/addAddV2inputsautoencoder/dense_422/Tanh:y:0*
T0*,
_output_shapes
:??????????2
autoencoder/add_53/add?
.autoencoder/dense_423/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_423_tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype020
.autoencoder/dense_423/Tensordot/ReadVariableOp?
$autoencoder/dense_423/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_423/Tensordot/axes?
$autoencoder/dense_423/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_423/Tensordot/free?
%autoencoder/dense_423/Tensordot/ShapeShapeautoencoder/add_53/add:z:0*
T0*
_output_shapes
:2'
%autoencoder/dense_423/Tensordot/Shape?
-autoencoder/dense_423/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_423/Tensordot/GatherV2/axis?
(autoencoder/dense_423/Tensordot/GatherV2GatherV2.autoencoder/dense_423/Tensordot/Shape:output:0-autoencoder/dense_423/Tensordot/free:output:06autoencoder/dense_423/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_423/Tensordot/GatherV2?
/autoencoder/dense_423/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_423/Tensordot/GatherV2_1/axis?
*autoencoder/dense_423/Tensordot/GatherV2_1GatherV2.autoencoder/dense_423/Tensordot/Shape:output:0-autoencoder/dense_423/Tensordot/axes:output:08autoencoder/dense_423/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_423/Tensordot/GatherV2_1?
%autoencoder/dense_423/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_423/Tensordot/Const?
$autoencoder/dense_423/Tensordot/ProdProd1autoencoder/dense_423/Tensordot/GatherV2:output:0.autoencoder/dense_423/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_423/Tensordot/Prod?
'autoencoder/dense_423/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_423/Tensordot/Const_1?
&autoencoder/dense_423/Tensordot/Prod_1Prod3autoencoder/dense_423/Tensordot/GatherV2_1:output:00autoencoder/dense_423/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_423/Tensordot/Prod_1?
+autoencoder/dense_423/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_423/Tensordot/concat/axis?
&autoencoder/dense_423/Tensordot/concatConcatV2-autoencoder/dense_423/Tensordot/free:output:0-autoencoder/dense_423/Tensordot/axes:output:04autoencoder/dense_423/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_423/Tensordot/concat?
%autoencoder/dense_423/Tensordot/stackPack-autoencoder/dense_423/Tensordot/Prod:output:0/autoencoder/dense_423/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_423/Tensordot/stack?
)autoencoder/dense_423/Tensordot/transpose	Transposeautoencoder/add_53/add:z:0/autoencoder/dense_423/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)autoencoder/dense_423/Tensordot/transpose?
'autoencoder/dense_423/Tensordot/ReshapeReshape-autoencoder/dense_423/Tensordot/transpose:y:0.autoencoder/dense_423/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_423/Tensordot/Reshape?
&autoencoder/dense_423/Tensordot/MatMulMatMul0autoencoder/dense_423/Tensordot/Reshape:output:06autoencoder/dense_423/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/dense_423/Tensordot/MatMul?
'autoencoder/dense_423/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'autoencoder/dense_423/Tensordot/Const_2?
-autoencoder/dense_423/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_423/Tensordot/concat_1/axis?
(autoencoder/dense_423/Tensordot/concat_1ConcatV21autoencoder/dense_423/Tensordot/GatherV2:output:00autoencoder/dense_423/Tensordot/Const_2:output:06autoencoder/dense_423/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_423/Tensordot/concat_1?
autoencoder/dense_423/TensordotReshape0autoencoder/dense_423/Tensordot/MatMul:product:01autoencoder/dense_423/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
autoencoder/dense_423/Tensordot?
,autoencoder/dense_423/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_423_biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02.
,autoencoder/dense_423/BiasAdd/ReadVariableOp?
autoencoder/dense_423/BiasAddBiasAdd(autoencoder/dense_423/Tensordot:output:04autoencoder/dense_423/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_423/BiasAdd?
 discriminator/dropout_1/IdentityIdentity&autoencoder/dense_423/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2"
 discriminator/dropout_1/Identity?
0discriminator/dense_424/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_424_tensordot_readvariableop_dense_424_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_424/Tensordot/ReadVariableOp?
&discriminator/dense_424/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_424/Tensordot/axes?
&discriminator/dense_424/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_424/Tensordot/free?
'discriminator/dense_424/Tensordot/ShapeShape)discriminator/dropout_1/Identity:output:0*
T0*
_output_shapes
:2)
'discriminator/dense_424/Tensordot/Shape?
/discriminator/dense_424/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_424/Tensordot/GatherV2/axis?
*discriminator/dense_424/Tensordot/GatherV2GatherV20discriminator/dense_424/Tensordot/Shape:output:0/discriminator/dense_424/Tensordot/free:output:08discriminator/dense_424/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_424/Tensordot/GatherV2?
1discriminator/dense_424/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_424/Tensordot/GatherV2_1/axis?
,discriminator/dense_424/Tensordot/GatherV2_1GatherV20discriminator/dense_424/Tensordot/Shape:output:0/discriminator/dense_424/Tensordot/axes:output:0:discriminator/dense_424/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_424/Tensordot/GatherV2_1?
'discriminator/dense_424/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_424/Tensordot/Const?
&discriminator/dense_424/Tensordot/ProdProd3discriminator/dense_424/Tensordot/GatherV2:output:00discriminator/dense_424/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_424/Tensordot/Prod?
)discriminator/dense_424/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_424/Tensordot/Const_1?
(discriminator/dense_424/Tensordot/Prod_1Prod5discriminator/dense_424/Tensordot/GatherV2_1:output:02discriminator/dense_424/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_424/Tensordot/Prod_1?
-discriminator/dense_424/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_424/Tensordot/concat/axis?
(discriminator/dense_424/Tensordot/concatConcatV2/discriminator/dense_424/Tensordot/free:output:0/discriminator/dense_424/Tensordot/axes:output:06discriminator/dense_424/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_424/Tensordot/concat?
'discriminator/dense_424/Tensordot/stackPack/discriminator/dense_424/Tensordot/Prod:output:01discriminator/dense_424/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_424/Tensordot/stack?
+discriminator/dense_424/Tensordot/transpose	Transpose)discriminator/dropout_1/Identity:output:01discriminator/dense_424/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_424/Tensordot/transpose?
)discriminator/dense_424/Tensordot/ReshapeReshape/discriminator/dense_424/Tensordot/transpose:y:00discriminator/dense_424/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_424/Tensordot/Reshape?
(discriminator/dense_424/Tensordot/MatMulMatMul2discriminator/dense_424/Tensordot/Reshape:output:08discriminator/dense_424/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_424/Tensordot/MatMul?
)discriminator/dense_424/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_424/Tensordot/Const_2?
/discriminator/dense_424/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_424/Tensordot/concat_1/axis?
*discriminator/dense_424/Tensordot/concat_1ConcatV23discriminator/dense_424/Tensordot/GatherV2:output:02discriminator/dense_424/Tensordot/Const_2:output:08discriminator/dense_424/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_424/Tensordot/concat_1?
!discriminator/dense_424/TensordotReshape2discriminator/dense_424/Tensordot/MatMul:product:03discriminator/dense_424/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_424/Tensordot?
.discriminator/dense_424/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_424_biasadd_readvariableop_dense_424_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_424/BiasAdd/ReadVariableOp?
discriminator/dense_424/BiasAddBiasAdd*discriminator/dense_424/Tensordot:output:06discriminator/dense_424/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_424/BiasAdd?
discriminator/dense_424/TanhTanh(discriminator/dense_424/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_424/Tanh?
0discriminator/dense_425/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_425_tensordot_readvariableop_dense_425_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_425/Tensordot/ReadVariableOp?
&discriminator/dense_425/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_425/Tensordot/axes?
&discriminator/dense_425/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_425/Tensordot/free?
'discriminator/dense_425/Tensordot/ShapeShape discriminator/dense_424/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_425/Tensordot/Shape?
/discriminator/dense_425/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_425/Tensordot/GatherV2/axis?
*discriminator/dense_425/Tensordot/GatherV2GatherV20discriminator/dense_425/Tensordot/Shape:output:0/discriminator/dense_425/Tensordot/free:output:08discriminator/dense_425/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_425/Tensordot/GatherV2?
1discriminator/dense_425/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_425/Tensordot/GatherV2_1/axis?
,discriminator/dense_425/Tensordot/GatherV2_1GatherV20discriminator/dense_425/Tensordot/Shape:output:0/discriminator/dense_425/Tensordot/axes:output:0:discriminator/dense_425/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_425/Tensordot/GatherV2_1?
'discriminator/dense_425/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_425/Tensordot/Const?
&discriminator/dense_425/Tensordot/ProdProd3discriminator/dense_425/Tensordot/GatherV2:output:00discriminator/dense_425/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_425/Tensordot/Prod?
)discriminator/dense_425/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_425/Tensordot/Const_1?
(discriminator/dense_425/Tensordot/Prod_1Prod5discriminator/dense_425/Tensordot/GatherV2_1:output:02discriminator/dense_425/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_425/Tensordot/Prod_1?
-discriminator/dense_425/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_425/Tensordot/concat/axis?
(discriminator/dense_425/Tensordot/concatConcatV2/discriminator/dense_425/Tensordot/free:output:0/discriminator/dense_425/Tensordot/axes:output:06discriminator/dense_425/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_425/Tensordot/concat?
'discriminator/dense_425/Tensordot/stackPack/discriminator/dense_425/Tensordot/Prod:output:01discriminator/dense_425/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_425/Tensordot/stack?
+discriminator/dense_425/Tensordot/transpose	Transpose discriminator/dense_424/Tanh:y:01discriminator/dense_425/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_425/Tensordot/transpose?
)discriminator/dense_425/Tensordot/ReshapeReshape/discriminator/dense_425/Tensordot/transpose:y:00discriminator/dense_425/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_425/Tensordot/Reshape?
(discriminator/dense_425/Tensordot/MatMulMatMul2discriminator/dense_425/Tensordot/Reshape:output:08discriminator/dense_425/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_425/Tensordot/MatMul?
)discriminator/dense_425/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_425/Tensordot/Const_2?
/discriminator/dense_425/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_425/Tensordot/concat_1/axis?
*discriminator/dense_425/Tensordot/concat_1ConcatV23discriminator/dense_425/Tensordot/GatherV2:output:02discriminator/dense_425/Tensordot/Const_2:output:08discriminator/dense_425/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_425/Tensordot/concat_1?
!discriminator/dense_425/TensordotReshape2discriminator/dense_425/Tensordot/MatMul:product:03discriminator/dense_425/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_425/Tensordot?
.discriminator/dense_425/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_425_biasadd_readvariableop_dense_425_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_425/BiasAdd/ReadVariableOp?
discriminator/dense_425/BiasAddBiasAdd*discriminator/dense_425/Tensordot:output:06discriminator/dense_425/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_425/BiasAdd?
discriminator/dense_425/TanhTanh(discriminator/dense_425/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_425/Tanh?
0discriminator/dense_426/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_426_tensordot_readvariableop_dense_426_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_426/Tensordot/ReadVariableOp?
&discriminator/dense_426/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_426/Tensordot/axes?
&discriminator/dense_426/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_426/Tensordot/free?
'discriminator/dense_426/Tensordot/ShapeShape discriminator/dense_425/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_426/Tensordot/Shape?
/discriminator/dense_426/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_426/Tensordot/GatherV2/axis?
*discriminator/dense_426/Tensordot/GatherV2GatherV20discriminator/dense_426/Tensordot/Shape:output:0/discriminator/dense_426/Tensordot/free:output:08discriminator/dense_426/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_426/Tensordot/GatherV2?
1discriminator/dense_426/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_426/Tensordot/GatherV2_1/axis?
,discriminator/dense_426/Tensordot/GatherV2_1GatherV20discriminator/dense_426/Tensordot/Shape:output:0/discriminator/dense_426/Tensordot/axes:output:0:discriminator/dense_426/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_426/Tensordot/GatherV2_1?
'discriminator/dense_426/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_426/Tensordot/Const?
&discriminator/dense_426/Tensordot/ProdProd3discriminator/dense_426/Tensordot/GatherV2:output:00discriminator/dense_426/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_426/Tensordot/Prod?
)discriminator/dense_426/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_426/Tensordot/Const_1?
(discriminator/dense_426/Tensordot/Prod_1Prod5discriminator/dense_426/Tensordot/GatherV2_1:output:02discriminator/dense_426/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_426/Tensordot/Prod_1?
-discriminator/dense_426/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_426/Tensordot/concat/axis?
(discriminator/dense_426/Tensordot/concatConcatV2/discriminator/dense_426/Tensordot/free:output:0/discriminator/dense_426/Tensordot/axes:output:06discriminator/dense_426/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_426/Tensordot/concat?
'discriminator/dense_426/Tensordot/stackPack/discriminator/dense_426/Tensordot/Prod:output:01discriminator/dense_426/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_426/Tensordot/stack?
+discriminator/dense_426/Tensordot/transpose	Transpose discriminator/dense_425/Tanh:y:01discriminator/dense_426/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_426/Tensordot/transpose?
)discriminator/dense_426/Tensordot/ReshapeReshape/discriminator/dense_426/Tensordot/transpose:y:00discriminator/dense_426/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_426/Tensordot/Reshape?
(discriminator/dense_426/Tensordot/MatMulMatMul2discriminator/dense_426/Tensordot/Reshape:output:08discriminator/dense_426/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_426/Tensordot/MatMul?
)discriminator/dense_426/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_426/Tensordot/Const_2?
/discriminator/dense_426/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_426/Tensordot/concat_1/axis?
*discriminator/dense_426/Tensordot/concat_1ConcatV23discriminator/dense_426/Tensordot/GatherV2:output:02discriminator/dense_426/Tensordot/Const_2:output:08discriminator/dense_426/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_426/Tensordot/concat_1?
!discriminator/dense_426/TensordotReshape2discriminator/dense_426/Tensordot/MatMul:product:03discriminator/dense_426/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_426/Tensordot?
.discriminator/dense_426/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_426_biasadd_readvariableop_dense_426_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_426/BiasAdd/ReadVariableOp?
discriminator/dense_426/BiasAddBiasAdd*discriminator/dense_426/Tensordot:output:06discriminator/dense_426/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_426/BiasAdd?
discriminator/dense_426/TanhTanh(discriminator/dense_426/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_426/Tanh?
0discriminator/dense_427/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_427_tensordot_readvariableop_dense_427_kernel*
_output_shapes
:	?*
dtype022
0discriminator/dense_427/Tensordot/ReadVariableOp?
&discriminator/dense_427/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_427/Tensordot/axes?
&discriminator/dense_427/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_427/Tensordot/free?
'discriminator/dense_427/Tensordot/ShapeShape discriminator/dense_426/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_427/Tensordot/Shape?
/discriminator/dense_427/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_427/Tensordot/GatherV2/axis?
*discriminator/dense_427/Tensordot/GatherV2GatherV20discriminator/dense_427/Tensordot/Shape:output:0/discriminator/dense_427/Tensordot/free:output:08discriminator/dense_427/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_427/Tensordot/GatherV2?
1discriminator/dense_427/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_427/Tensordot/GatherV2_1/axis?
,discriminator/dense_427/Tensordot/GatherV2_1GatherV20discriminator/dense_427/Tensordot/Shape:output:0/discriminator/dense_427/Tensordot/axes:output:0:discriminator/dense_427/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_427/Tensordot/GatherV2_1?
'discriminator/dense_427/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_427/Tensordot/Const?
&discriminator/dense_427/Tensordot/ProdProd3discriminator/dense_427/Tensordot/GatherV2:output:00discriminator/dense_427/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_427/Tensordot/Prod?
)discriminator/dense_427/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_427/Tensordot/Const_1?
(discriminator/dense_427/Tensordot/Prod_1Prod5discriminator/dense_427/Tensordot/GatherV2_1:output:02discriminator/dense_427/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_427/Tensordot/Prod_1?
-discriminator/dense_427/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_427/Tensordot/concat/axis?
(discriminator/dense_427/Tensordot/concatConcatV2/discriminator/dense_427/Tensordot/free:output:0/discriminator/dense_427/Tensordot/axes:output:06discriminator/dense_427/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_427/Tensordot/concat?
'discriminator/dense_427/Tensordot/stackPack/discriminator/dense_427/Tensordot/Prod:output:01discriminator/dense_427/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_427/Tensordot/stack?
+discriminator/dense_427/Tensordot/transpose	Transpose discriminator/dense_426/Tanh:y:01discriminator/dense_427/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_427/Tensordot/transpose?
)discriminator/dense_427/Tensordot/ReshapeReshape/discriminator/dense_427/Tensordot/transpose:y:00discriminator/dense_427/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_427/Tensordot/Reshape?
(discriminator/dense_427/Tensordot/MatMulMatMul2discriminator/dense_427/Tensordot/Reshape:output:08discriminator/dense_427/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(discriminator/dense_427/Tensordot/MatMul?
)discriminator/dense_427/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)discriminator/dense_427/Tensordot/Const_2?
/discriminator/dense_427/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_427/Tensordot/concat_1/axis?
*discriminator/dense_427/Tensordot/concat_1ConcatV23discriminator/dense_427/Tensordot/GatherV2:output:02discriminator/dense_427/Tensordot/Const_2:output:08discriminator/dense_427/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_427/Tensordot/concat_1?
!discriminator/dense_427/TensordotReshape2discriminator/dense_427/Tensordot/MatMul:product:03discriminator/dense_427/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2#
!discriminator/dense_427/Tensordot?
.discriminator/dense_427/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_427_biasadd_readvariableop_dense_427_bias*
_output_shapes
:*
dtype020
.discriminator/dense_427/BiasAdd/ReadVariableOp?
discriminator/dense_427/BiasAddBiasAdd*discriminator/dense_427/Tensordot:output:06discriminator/dense_427/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2!
discriminator/dense_427/BiasAdd?
discriminator/dense_427/TanhTanh(discriminator/dense_427/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
discriminator/dense_427/Tanh?
discriminator/flatten_52/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2 
discriminator/flatten_52/Const?
 discriminator/flatten_52/ReshapeReshape discriminator/dense_427/Tanh:y:0'discriminator/flatten_52/Const:output:0*
T0*'
_output_shapes
:?????????2"
 discriminator/flatten_52/Reshape?
-discriminator/dense_428/MatMul/ReadVariableOpReadVariableOp>discriminator_dense_428_matmul_readvariableop_dense_428_kernel*
_output_shapes

:*
dtype02/
-discriminator/dense_428/MatMul/ReadVariableOp?
discriminator/dense_428/MatMulMatMul)discriminator/flatten_52/Reshape:output:05discriminator/dense_428/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
discriminator/dense_428/MatMul?
.discriminator/dense_428/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_428_biasadd_readvariableop_dense_428_bias*
_output_shapes
:*
dtype020
.discriminator/dense_428/BiasAdd/ReadVariableOp?
discriminator/dense_428/BiasAddBiasAdd(discriminator/dense_428/MatMul:product:06discriminator/dense_428/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
discriminator/dense_428/BiasAdd?
discriminator/dense_428/SigmoidSigmoid(discriminator/dense_428/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2!
discriminator/dense_428/Sigmoid?
IdentityIdentity#discriminator/dense_428/Sigmoid:y:0-^autoencoder/dense_421/BiasAdd/ReadVariableOp/^autoencoder/dense_421/Tensordot/ReadVariableOp-^autoencoder/dense_422/BiasAdd/ReadVariableOp/^autoencoder/dense_422/Tensordot/ReadVariableOp-^autoencoder/dense_423/BiasAdd/ReadVariableOp/^autoencoder/dense_423/Tensordot/ReadVariableOp/^discriminator/dense_424/BiasAdd/ReadVariableOp1^discriminator/dense_424/Tensordot/ReadVariableOp/^discriminator/dense_425/BiasAdd/ReadVariableOp1^discriminator/dense_425/Tensordot/ReadVariableOp/^discriminator/dense_426/BiasAdd/ReadVariableOp1^discriminator/dense_426/Tensordot/ReadVariableOp/^discriminator/dense_427/BiasAdd/ReadVariableOp1^discriminator/dense_427/Tensordot/ReadVariableOp/^discriminator/dense_428/BiasAdd/ReadVariableOp.^discriminator/dense_428/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::2\
,autoencoder/dense_421/BiasAdd/ReadVariableOp,autoencoder/dense_421/BiasAdd/ReadVariableOp2`
.autoencoder/dense_421/Tensordot/ReadVariableOp.autoencoder/dense_421/Tensordot/ReadVariableOp2\
,autoencoder/dense_422/BiasAdd/ReadVariableOp,autoencoder/dense_422/BiasAdd/ReadVariableOp2`
.autoencoder/dense_422/Tensordot/ReadVariableOp.autoencoder/dense_422/Tensordot/ReadVariableOp2\
,autoencoder/dense_423/BiasAdd/ReadVariableOp,autoencoder/dense_423/BiasAdd/ReadVariableOp2`
.autoencoder/dense_423/Tensordot/ReadVariableOp.autoencoder/dense_423/Tensordot/ReadVariableOp2`
.discriminator/dense_424/BiasAdd/ReadVariableOp.discriminator/dense_424/BiasAdd/ReadVariableOp2d
0discriminator/dense_424/Tensordot/ReadVariableOp0discriminator/dense_424/Tensordot/ReadVariableOp2`
.discriminator/dense_425/BiasAdd/ReadVariableOp.discriminator/dense_425/BiasAdd/ReadVariableOp2d
0discriminator/dense_425/Tensordot/ReadVariableOp0discriminator/dense_425/Tensordot/ReadVariableOp2`
.discriminator/dense_426/BiasAdd/ReadVariableOp.discriminator/dense_426/BiasAdd/ReadVariableOp2d
0discriminator/dense_426/Tensordot/ReadVariableOp0discriminator/dense_426/Tensordot/ReadVariableOp2`
.discriminator/dense_427/BiasAdd/ReadVariableOp.discriminator/dense_427/BiasAdd/ReadVariableOp2d
0discriminator/dense_427/Tensordot/ReadVariableOp0discriminator/dense_427/Tensordot/ReadVariableOp2`
.discriminator/dense_428/BiasAdd/ReadVariableOp.discriminator/dense_428/BiasAdd/ReadVariableOp2^
-discriminator/dense_428/MatMul/ReadVariableOp-discriminator/dense_428/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_61000625
inputs_0
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0dense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609995822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?
?
,__inference_model_975_layer_call_fn_61000241

inputs
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_biasdense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_975_layer_call_and_return_conditional_losses_609997442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_52_layer_call_and_return_conditional_losses_60999283

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_425_layer_call_and_return_conditional_losses_61001127

inputs-
)tensordot_readvariableop_dense_425_kernel)
%biasadd_readvariableop_dense_425_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_425_kernel* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_425_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_dense_428_layer_call_and_return_conditional_losses_60999302

inputs*
&matmul_readvariableop_dense_428_kernel)
%biasadd_readvariableop_dense_428_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp&matmul_readvariableop_dense_428_kernel*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_428_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_422_layer_call_and_return_conditional_losses_61000975

inputs-
)tensordot_readvariableop_dense_422_kernel)
%biasadd_readvariableop_dense_422_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_422_kernel*
_output_shapes
:	 ?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_422_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_60999071
input_13
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13dense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609990622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
?	
?
0__inference_discriminator_layer_call_fn_61000891

inputs
dense_424_kernel
dense_424_bias
dense_425_kernel
dense_425_bias
dense_426_kernel
dense_426_bias
dense_427_kernel
dense_427_bias
dense_428_kernel
dense_428_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_424_kerneldense_424_biasdense_425_kerneldense_425_biasdense_426_kerneldense_426_biasdense_427_kerneldense_427_biasdense_428_kerneldense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609993602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_flatten_52_layer_call_fn_61001221

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_52_layer_call_and_return_conditional_losses_609992832
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_975_layer_call_and_return_conditional_losses_60999654
input_13 
autoencoder_dense_421_kernel
autoencoder_dense_421_bias 
autoencoder_dense_422_kernel
autoencoder_dense_422_bias 
autoencoder_dense_423_kernel
autoencoder_dense_423_bias"
discriminator_dense_424_kernel 
discriminator_dense_424_bias"
discriminator_dense_425_kernel 
discriminator_dense_425_bias"
discriminator_dense_426_kernel 
discriminator_dense_426_bias"
discriminator_dense_427_kernel 
discriminator_dense_427_bias"
discriminator_dense_428_kernel 
discriminator_dense_428_bias
identity??#autoencoder/StatefulPartitionedCall?%discriminator/StatefulPartitionedCall?
#autoencoder/StatefulPartitionedCallStatefulPartitionedCallinput_13autoencoder_dense_421_kernelautoencoder_dense_421_biasautoencoder_dense_422_kernelautoencoder_dense_422_biasautoencoder_dense_423_kernelautoencoder_dense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609994972%
#autoencoder/StatefulPartitionedCall?
%discriminator/StatefulPartitionedCallStatefulPartitionedCall,autoencoder/StatefulPartitionedCall:output:0discriminator_dense_424_kerneldiscriminator_dense_424_biasdiscriminator_dense_425_kerneldiscriminator_dense_425_biasdiscriminator_dense_426_kerneldiscriminator_dense_426_biasdiscriminator_dense_427_kerneldiscriminator_dense_427_biasdiscriminator_dense_428_kerneldiscriminator_dense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609993602'
%discriminator/StatefulPartitionedCall?
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0$^autoencoder/StatefulPartitionedCall&^discriminator/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::2J
#autoencoder/StatefulPartitionedCall#autoencoder/StatefulPartitionedCall2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
?
?
,__inference_dense_423_layer_call_fn_61001031

inputs
dense_423_kernel
dense_423_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_423_kerneldense_423_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_423_layer_call_and_return_conditional_losses_609989932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_dense_428_layer_call_and_return_conditional_losses_61001232

inputs*
&matmul_readvariableop_dense_428_kernel)
%biasadd_readvariableop_dense_428_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp&matmul_readvariableop_dense_428_kernel*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_428_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_60999046
input_13
dense_421_kernel
dense_421_bias
dense_422_kernel
dense_422_bias
dense_423_kernel
dense_423_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13dense_421_kerneldense_421_biasdense_422_kerneldense_422_biasdense_423_kerneldense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609990372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_13
?#
?
K__inference_discriminator_layer_call_and_return_conditional_losses_60999336
input_14
dense_424_dense_424_kernel
dense_424_dense_424_bias
dense_425_dense_425_kernel
dense_425_dense_425_bias
dense_426_dense_426_kernel
dense_426_dense_426_bias
dense_427_dense_427_kernel
dense_427_dense_427_bias
dense_428_dense_428_kernel
dense_428_dense_428_bias
identity??!dense_424/StatefulPartitionedCall?!dense_425/StatefulPartitionedCall?!dense_426/StatefulPartitionedCall?!dense_427/StatefulPartitionedCall?!dense_428/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCallinput_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_609990922
dropout_1/PartitionedCall?
!dense_424/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_424_dense_424_kerneldense_424_dense_424_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_424_layer_call_and_return_conditional_losses_609991362#
!dense_424/StatefulPartitionedCall?
!dense_425/StatefulPartitionedCallStatefulPartitionedCall*dense_424/StatefulPartitionedCall:output:0dense_425_dense_425_kerneldense_425_dense_425_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_425_layer_call_and_return_conditional_losses_609991792#
!dense_425/StatefulPartitionedCall?
!dense_426/StatefulPartitionedCallStatefulPartitionedCall*dense_425/StatefulPartitionedCall:output:0dense_426_dense_426_kerneldense_426_dense_426_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_426_layer_call_and_return_conditional_losses_609992222#
!dense_426/StatefulPartitionedCall?
!dense_427/StatefulPartitionedCallStatefulPartitionedCall*dense_426/StatefulPartitionedCall:output:0dense_427_dense_427_kerneldense_427_dense_427_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_427_layer_call_and_return_conditional_losses_609992652#
!dense_427/StatefulPartitionedCall?
flatten_52/PartitionedCallPartitionedCall*dense_427/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_52_layer_call_and_return_conditional_losses_609992832
flatten_52/PartitionedCall?
!dense_428/StatefulPartitionedCallStatefulPartitionedCall#flatten_52/PartitionedCall:output:0dense_428_dense_428_kerneldense_428_dense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_428_layer_call_and_return_conditional_losses_609993022#
!dense_428/StatefulPartitionedCall?
IdentityIdentity*dense_428/StatefulPartitionedCall:output:0"^dense_424/StatefulPartitionedCall"^dense_425/StatefulPartitionedCall"^dense_426/StatefulPartitionedCall"^dense_427/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:??????????::::::::::2F
!dense_424/StatefulPartitionedCall!dense_424/StatefulPartitionedCall2F
!dense_425/StatefulPartitionedCall!dense_425/StatefulPartitionedCall2F
!dense_426/StatefulPartitionedCall!dense_426/StatefulPartitionedCall2F
!dense_427/StatefulPartitionedCall!dense_427/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_14
?
?
G__inference_dense_423_layer_call_and_return_conditional_losses_60998993

inputs-
)tensordot_readvariableop_dense_423_kernel)
%biasadd_readvariableop_dense_423_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_423_kernel* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_423_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_425_layer_call_and_return_conditional_losses_60999179

inputs-
)tensordot_readvariableop_dense_425_kernel)
%biasadd_readvariableop_dense_425_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_425_kernel* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_425_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_60999092

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_model_975_layer_call_and_return_conditional_losses_60999701

inputs 
autoencoder_dense_421_kernel
autoencoder_dense_421_bias 
autoencoder_dense_422_kernel
autoencoder_dense_422_bias 
autoencoder_dense_423_kernel
autoencoder_dense_423_bias"
discriminator_dense_424_kernel 
discriminator_dense_424_bias"
discriminator_dense_425_kernel 
discriminator_dense_425_bias"
discriminator_dense_426_kernel 
discriminator_dense_426_bias"
discriminator_dense_427_kernel 
discriminator_dense_427_bias"
discriminator_dense_428_kernel 
discriminator_dense_428_bias
identity??#autoencoder/StatefulPartitionedCall?%discriminator/StatefulPartitionedCall?
#autoencoder/StatefulPartitionedCallStatefulPartitionedCallinputsautoencoder_dense_421_kernelautoencoder_dense_421_biasautoencoder_dense_422_kernelautoencoder_dense_422_biasautoencoder_dense_423_kernelautoencoder_dense_423_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609994972%
#autoencoder/StatefulPartitionedCall?
%discriminator/StatefulPartitionedCallStatefulPartitionedCall,autoencoder/StatefulPartitionedCall:output:0discriminator_dense_424_kerneldiscriminator_dense_424_biasdiscriminator_dense_425_kerneldiscriminator_dense_425_biasdiscriminator_dense_426_kerneldiscriminator_dense_426_biasdiscriminator_dense_427_kerneldiscriminator_dense_427_biasdiscriminator_dense_428_kerneldiscriminator_dense_428_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609993602'
%discriminator/StatefulPartitionedCall?
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0$^autoencoder/StatefulPartitionedCall&^discriminator/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????::::::::::::::::2J
#autoencoder/StatefulPartitionedCall#autoencoder/StatefulPartitionedCall2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
B
input_136
serving_default_input_13:0??????????A
discriminator0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?d
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?b
_tf_keras_network?b{"class_name": "Functional", "name": "model_975", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_975", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}, "name": "input_13", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}, "name": "input_13", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_421", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_421", "inbound_nodes": [[["input_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_422", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_422", "inbound_nodes": [[["dense_421", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_53", "trainable": true, "dtype": "float32"}, "name": "add_53", "inbound_nodes": [[["input_13", 0, 0, {}], ["dense_422", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_423", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_423", "inbound_nodes": [[["add_53", 0, 0, {}]]]}], "input_layers": [["input_13", 0, 0]], "output_layers": [["dense_423", 0, 0]]}, "name": "autoencoder", "inbound_nodes": [[["input_13", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}, "name": "input_14", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["input_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_424", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_424", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_425", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_425", "inbound_nodes": [[["dense_424", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_426", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_426", "inbound_nodes": [[["dense_425", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_427", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_427", "inbound_nodes": [[["dense_426", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_52", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_52", "inbound_nodes": [[["dense_427", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_428", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_428", "inbound_nodes": [[["flatten_52", 0, 0, {}]]]}], "input_layers": [["input_14", 0, 0]], "output_layers": [["dense_428", 0, 0]]}, "name": "discriminator", "inbound_nodes": [[["autoencoder", 1, 0, {}]]]}], "input_layers": [["input_13", 0, 0]], "output_layers": [["discriminator", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 513]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_975", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}, "name": "input_13", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}, "name": "input_13", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_421", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_421", "inbound_nodes": [[["input_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_422", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_422", "inbound_nodes": [[["dense_421", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_53", "trainable": true, "dtype": "float32"}, "name": "add_53", "inbound_nodes": [[["input_13", 0, 0, {}], ["dense_422", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_423", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_423", "inbound_nodes": [[["add_53", 0, 0, {}]]]}], "input_layers": [["input_13", 0, 0]], "output_layers": [["dense_423", 0, 0]]}, "name": "autoencoder", "inbound_nodes": [[["input_13", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}, "name": "input_14", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["input_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_424", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_424", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_425", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_425", "inbound_nodes": [[["dense_424", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_426", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_426", "inbound_nodes": [[["dense_425", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_427", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_427", "inbound_nodes": [[["dense_426", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_52", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_52", "inbound_nodes": [[["dense_427", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_428", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_428", "inbound_nodes": [[["flatten_52", 0, 0, {}]]]}], "input_layers": [["input_14", 0, 0]], "output_layers": [["dense_428", 0, 0]]}, "name": "discriminator", "inbound_nodes": [[["autoencoder", 1, 0, {}]]]}], "input_layers": [["input_13", 0, 0]], "output_layers": [["discriminator", 1, 0]]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_13", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}}
?)
layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?&
_tf_keras_network?&{"class_name": "Functional", "name": "autoencoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}, "name": "input_13", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_421", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_421", "inbound_nodes": [[["input_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_422", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_422", "inbound_nodes": [[["dense_421", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_53", "trainable": true, "dtype": "float32"}, "name": "add_53", "inbound_nodes": [[["input_13", 0, 0, {}], ["dense_422", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_423", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_423", "inbound_nodes": [[["add_53", 0, 0, {}]]]}], "input_layers": [["input_13", 0, 0]], "output_layers": [["dense_423", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 513]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}, "name": "input_13", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_421", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_421", "inbound_nodes": [[["input_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_422", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_422", "inbound_nodes": [[["dense_421", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_53", "trainable": true, "dtype": "float32"}, "name": "add_53", "inbound_nodes": [[["input_13", 0, 0, {}], ["dense_422", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_423", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_423", "inbound_nodes": [[["add_53", 0, 0, {}]]]}], "input_layers": [["input_13", 0, 0]], "output_layers": [["dense_423", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?=
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?:
_tf_keras_network?:{"class_name": "Functional", "name": "discriminator", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}, "name": "input_14", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["input_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_424", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_424", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_425", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_425", "inbound_nodes": [[["dense_424", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_426", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_426", "inbound_nodes": [[["dense_425", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_427", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_427", "inbound_nodes": [[["dense_426", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_52", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_52", "inbound_nodes": [[["dense_427", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_428", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_428", "inbound_nodes": [[["flatten_52", 0, 0, {}]]]}], "input_layers": [["input_14", 0, 0]], "output_layers": [["dense_428", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 513]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}, "name": "input_14", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["input_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_424", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_424", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_425", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_425", "inbound_nodes": [[["dense_424", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_426", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_426", "inbound_nodes": [[["dense_425", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_427", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_427", "inbound_nodes": [[["dense_426", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_52", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_52", "inbound_nodes": [[["dense_427", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_428", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_428", "inbound_nodes": [[["flatten_52", 0, 0, {}]]]}], "input_layers": [["input_14", 0, 0]], "output_layers": [["dense_428", 0, 0]]}}, "training_config": {"loss": "bce", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
 iter

!beta_1

"beta_2
	#decay
$learning_rate%m?&m?'m?(m?)m?*m?%v?&v?'v?(v?)v?*v?"
	optimizer
 "
trackable_list_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
?
%0
&1
'2
(3
)4
*5
+6
,7
-8
.9
/10
011
112
213
314
415"
trackable_list_wrapper
?
regularization_losses
trainable_variables

5layers
6layer_regularization_losses
7metrics
8layer_metrics
9non_trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

%kernel
&bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_421", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_421", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 513}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 513]}}
?

'kernel
(bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_422", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_422", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
?
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_53", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 3, 513]}, {"class_name": "TensorShape", "items": [null, 3, 513]}]}
?

)kernel
*bias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_423", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_423", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 513}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 513]}}
"
	optimizer
 "
trackable_list_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
?
regularization_losses
trainable_variables

Jlayers
Klayer_regularization_losses
Lmetrics
Mlayer_metrics
Nnon_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_14", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}}
?
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

+kernel
,bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_424", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_424", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 513}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 513]}}
?

-kernel
.bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_425", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_425", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 512]}}
?

/kernel
0bias
[regularization_losses
\trainable_variables
]	variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_426", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_426", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 256]}}
?

1kernel
2bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_427", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_427", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 128]}}
?
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_52", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_52", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

3kernel
4bias
gregularization_losses
htrainable_variables
i	variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_428", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_428", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
kiter

lbeta_1

mbeta_2
	ndecay
olearning_rate+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
+0
,1
-2
.3
/4
05
16
27
38
49"
trackable_list_wrapper
?
regularization_losses
trainable_variables

players
qlayer_regularization_losses
rmetrics
slayer_metrics
tnon_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	 (2training_334/Adam/iter
":  (2training_334/Adam/beta_1
":  (2training_334/Adam/beta_2
!: (2training_334/Adam/decay
):' (2training_334/Adam/learning_rate
#:!	? 2dense_421/kernel
: 2dense_421/bias
#:!	 ?2dense_422/kernel
:?2dense_422/bias
$:"
??2dense_423/kernel
:?2dense_423/bias
$:"
??2dense_424/kernel
:?2dense_424/bias
$:"
??2dense_425/kernel
:?2dense_425/bias
$:"
??2dense_426/kernel
:?2dense_426/bias
#:!	?2dense_427/kernel
:2dense_427/bias
": 2dense_428/kernel
:2dense_428/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
'
u0"
trackable_list_wrapper
 "
trackable_dict_wrapper
f
+0
,1
-2
.3
/4
05
16
27
38
49"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
:regularization_losses
;trainable_variables
vlayer_regularization_losses

wlayers
xmetrics
ylayer_metrics
znon_trainable_variables
<	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
>regularization_losses
?trainable_variables
{layer_regularization_losses

|layers
}metrics
~layer_metrics
non_trainable_variables
@	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bregularization_losses
Ctrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
D	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
Fregularization_losses
Gtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
H	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
C
0

1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
?
Oregularization_losses
Ptrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
Q	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
Sregularization_losses
Ttrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
U	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
Wregularization_losses
Xtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
Y	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
[regularization_losses
\trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
]	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
_regularization_losses
`trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
a	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
cregularization_losses
dtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
e	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
gregularization_losses
htrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
i	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	 (2training_320/Adam/iter
":  (2training_320/Adam/beta_1
":  (2training_320/Adam/beta_2
!: (2training_320/Adam/decay
):' (2training_320/Adam/learning_rate
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
f
+0
,1
-2
.3
/4
05
16
27
38
49"
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
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
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
10
21"
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
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2	total_346
:  (2	count_346
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2	total_330
:  (2	count_330
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2	total_331
:  (2	count_331
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
5:3	? 2$training_334/Adam/dense_421/kernel/m
.:, 2"training_334/Adam/dense_421/bias/m
5:3	 ?2$training_334/Adam/dense_422/kernel/m
/:-?2"training_334/Adam/dense_422/bias/m
6:4
??2$training_334/Adam/dense_423/kernel/m
/:-?2"training_334/Adam/dense_423/bias/m
5:3	? 2$training_334/Adam/dense_421/kernel/v
.:, 2"training_334/Adam/dense_421/bias/v
5:3	 ?2$training_334/Adam/dense_422/kernel/v
/:-?2"training_334/Adam/dense_422/bias/v
6:4
??2$training_334/Adam/dense_423/kernel/v
/:-?2"training_334/Adam/dense_423/bias/v
6:4
??2$training_320/Adam/dense_424/kernel/m
/:-?2"training_320/Adam/dense_424/bias/m
6:4
??2$training_320/Adam/dense_425/kernel/m
/:-?2"training_320/Adam/dense_425/bias/m
6:4
??2$training_320/Adam/dense_426/kernel/m
/:-?2"training_320/Adam/dense_426/bias/m
5:3	?2$training_320/Adam/dense_427/kernel/m
.:,2"training_320/Adam/dense_427/bias/m
4:22$training_320/Adam/dense_428/kernel/m
.:,2"training_320/Adam/dense_428/bias/m
6:4
??2$training_320/Adam/dense_424/kernel/v
/:-?2"training_320/Adam/dense_424/bias/v
6:4
??2$training_320/Adam/dense_425/kernel/v
/:-?2"training_320/Adam/dense_425/bias/v
6:4
??2$training_320/Adam/dense_426/kernel/v
/:-?2"training_320/Adam/dense_426/bias/v
5:3	?2$training_320/Adam/dense_427/kernel/v
.:,2"training_320/Adam/dense_427/bias/v
4:22$training_320/Adam/dense_428/kernel/v
.:,2"training_320/Adam/dense_428/bias/v
?2?
G__inference_model_975_layer_call_and_return_conditional_losses_61000199
G__inference_model_975_layer_call_and_return_conditional_losses_60999996
G__inference_model_975_layer_call_and_return_conditional_losses_60999654
G__inference_model_975_layer_call_and_return_conditional_losses_60999676?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_model_975_layer_call_fn_60999763
,__inference_model_975_layer_call_fn_61000241
,__inference_model_975_layer_call_fn_61000220
,__inference_model_975_layer_call_fn_60999720?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_60998858?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *,?)
'?$
input_13??????????
?2?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000411
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000518
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000603
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000326
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999020
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999006?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_autoencoder_layer_call_fn_60999046
.__inference_autoencoder_layer_call_fn_60999071
.__inference_autoencoder_layer_call_fn_61000614
.__inference_autoencoder_layer_call_fn_61000625
.__inference_autoencoder_layer_call_fn_61000433
.__inference_autoencoder_layer_call_fn_61000422?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_discriminator_layer_call_and_return_conditional_losses_61000876
K__inference_discriminator_layer_call_and_return_conditional_losses_61000754
K__inference_discriminator_layer_call_and_return_conditional_losses_60999336
K__inference_discriminator_layer_call_and_return_conditional_losses_60999315?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_discriminator_layer_call_fn_61000891
0__inference_discriminator_layer_call_fn_60999409
0__inference_discriminator_layer_call_fn_61000906
0__inference_discriminator_layer_call_fn_60999373?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_signature_wrapper_60999786input_13"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_421_layer_call_and_return_conditional_losses_61000937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_421_layer_call_fn_61000944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_422_layer_call_and_return_conditional_losses_61000975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_422_layer_call_fn_61000982?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_add_53_layer_call_and_return_conditional_losses_61000988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_add_53_layer_call_fn_61000994?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_423_layer_call_and_return_conditional_losses_61001024?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_423_layer_call_fn_61001031?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dropout_1_layer_call_and_return_conditional_losses_61001043
G__inference_dropout_1_layer_call_and_return_conditional_losses_61001048?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_1_layer_call_fn_61001053
,__inference_dropout_1_layer_call_fn_61001058?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dense_424_layer_call_and_return_conditional_losses_61001089?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_424_layer_call_fn_61001096?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_425_layer_call_and_return_conditional_losses_61001127?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_425_layer_call_fn_61001134?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_426_layer_call_and_return_conditional_losses_61001165?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_426_layer_call_fn_61001172?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_427_layer_call_and_return_conditional_losses_61001203?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_427_layer_call_fn_61001210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_52_layer_call_and_return_conditional_losses_61001216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_52_layer_call_fn_61001221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_428_layer_call_and_return_conditional_losses_61001232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_428_layer_call_fn_61001239?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_60998858?%&'()*+,-./012346?3
,?)
'?$
input_13??????????
? "=?:
8
discriminator'?$
discriminator??????????
D__inference_add_53_layer_call_and_return_conditional_losses_61000988?d?a
Z?W
U?R
'?$
inputs/0??????????
'?$
inputs/1??????????
? "*?'
 ?
0??????????
? ?
)__inference_add_53_layer_call_fn_61000994?d?a
Z?W
U?R
'?$
inputs/0??????????
'?$
inputs/1??????????
? "????????????
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999006t%&'()*>?;
4?1
'?$
input_13??????????
p

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60999020t%&'()*>?;
4?1
'?$
input_13??????????
p 

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000326r%&'()*<?9
2?/
%?"
inputs??????????
p

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000411r%&'()*<?9
2?/
%?"
inputs??????????
p 

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000518y%&'()*C?@
9?6
,?)
'?$
inputs/0??????????
p

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_61000603y%&'()*C?@
9?6
,?)
'?$
inputs/0??????????
p 

 
? "*?'
 ?
0??????????
? ?
.__inference_autoencoder_layer_call_fn_60999046g%&'()*>?;
4?1
'?$
input_13??????????
p

 
? "????????????
.__inference_autoencoder_layer_call_fn_60999071g%&'()*>?;
4?1
'?$
input_13??????????
p 

 
? "????????????
.__inference_autoencoder_layer_call_fn_61000422e%&'()*<?9
2?/
%?"
inputs??????????
p

 
? "????????????
.__inference_autoencoder_layer_call_fn_61000433e%&'()*<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
.__inference_autoencoder_layer_call_fn_61000614l%&'()*C?@
9?6
,?)
'?$
inputs/0??????????
p

 
? "????????????
.__inference_autoencoder_layer_call_fn_61000625l%&'()*C?@
9?6
,?)
'?$
inputs/0??????????
p 

 
? "????????????
G__inference_dense_421_layer_call_and_return_conditional_losses_61000937e%&4?1
*?'
%?"
inputs??????????
? ")?&
?
0????????? 
? ?
,__inference_dense_421_layer_call_fn_61000944X%&4?1
*?'
%?"
inputs??????????
? "?????????? ?
G__inference_dense_422_layer_call_and_return_conditional_losses_61000975e'(3?0
)?&
$?!
inputs????????? 
? "*?'
 ?
0??????????
? ?
,__inference_dense_422_layer_call_fn_61000982X'(3?0
)?&
$?!
inputs????????? 
? "????????????
G__inference_dense_423_layer_call_and_return_conditional_losses_61001024f)*4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_423_layer_call_fn_61001031Y)*4?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_424_layer_call_and_return_conditional_losses_61001089f+,4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_424_layer_call_fn_61001096Y+,4?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_425_layer_call_and_return_conditional_losses_61001127f-.4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_425_layer_call_fn_61001134Y-.4?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_426_layer_call_and_return_conditional_losses_61001165f/04?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_426_layer_call_fn_61001172Y/04?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_427_layer_call_and_return_conditional_losses_61001203e124?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
,__inference_dense_427_layer_call_fn_61001210X124?1
*?'
%?"
inputs??????????
? "???????????
G__inference_dense_428_layer_call_and_return_conditional_losses_61001232\34/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_428_layer_call_fn_61001239O34/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_discriminator_layer_call_and_return_conditional_losses_60999315s
+,-./01234>?;
4?1
'?$
input_14??????????
p

 
? "%?"
?
0?????????
? ?
K__inference_discriminator_layer_call_and_return_conditional_losses_60999336s
+,-./01234>?;
4?1
'?$
input_14??????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_discriminator_layer_call_and_return_conditional_losses_61000754q
+,-./01234<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????
? ?
K__inference_discriminator_layer_call_and_return_conditional_losses_61000876q
+,-./01234<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
0__inference_discriminator_layer_call_fn_60999373f
+,-./01234>?;
4?1
'?$
input_14??????????
p

 
? "???????????
0__inference_discriminator_layer_call_fn_60999409f
+,-./01234>?;
4?1
'?$
input_14??????????
p 

 
? "???????????
0__inference_discriminator_layer_call_fn_61000891d
+,-./01234<?9
2?/
%?"
inputs??????????
p

 
? "???????????
0__inference_discriminator_layer_call_fn_61000906d
+,-./01234<?9
2?/
%?"
inputs??????????
p 

 
? "???????????
G__inference_dropout_1_layer_call_and_return_conditional_losses_61001043f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
G__inference_dropout_1_layer_call_and_return_conditional_losses_61001048f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
,__inference_dropout_1_layer_call_fn_61001053Y8?5
.?+
%?"
inputs??????????
p
? "????????????
,__inference_dropout_1_layer_call_fn_61001058Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
H__inference_flatten_52_layer_call_and_return_conditional_losses_61001216\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? ?
-__inference_flatten_52_layer_call_fn_61001221O3?0
)?&
$?!
inputs?????????
? "???????????
G__inference_model_975_layer_call_and_return_conditional_losses_60999654y%&'()*+,-./01234>?;
4?1
'?$
input_13??????????
p

 
? "%?"
?
0?????????
? ?
G__inference_model_975_layer_call_and_return_conditional_losses_60999676y%&'()*+,-./01234>?;
4?1
'?$
input_13??????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_975_layer_call_and_return_conditional_losses_60999996w%&'()*+,-./01234<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????
? ?
G__inference_model_975_layer_call_and_return_conditional_losses_61000199w%&'()*+,-./01234<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_model_975_layer_call_fn_60999720l%&'()*+,-./01234>?;
4?1
'?$
input_13??????????
p

 
? "???????????
,__inference_model_975_layer_call_fn_60999763l%&'()*+,-./01234>?;
4?1
'?$
input_13??????????
p 

 
? "???????????
,__inference_model_975_layer_call_fn_61000220j%&'()*+,-./01234<?9
2?/
%?"
inputs??????????
p

 
? "???????????
,__inference_model_975_layer_call_fn_61000241j%&'()*+,-./01234<?9
2?/
%?"
inputs??????????
p 

 
? "???????????
&__inference_signature_wrapper_60999786?%&'()*+,-./01234B??
? 
8?5
3
input_13'?$
input_13??????????"=?:
8
discriminator'?$
discriminator?????????