??%
??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??!
?
training_318/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *'
shared_nametraining_318/Adam/iter
y
*training_318/Adam/iter/Read/ReadVariableOpReadVariableOptraining_318/Adam/iter*
_output_shapes
: *
dtype0	
?
training_318/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_318/Adam/beta_1
}
,training_318/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_318/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_318/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_318/Adam/beta_2
}
,training_318/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_318/Adam/beta_2*
_output_shapes
: *
dtype0
?
training_318/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametraining_318/Adam/decay
{
+training_318/Adam/decay/Read/ReadVariableOpReadVariableOptraining_318/Adam/decay*
_output_shapes
: *
dtype0
?
training_318/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training_318/Adam/learning_rate
?
3training_318/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_318/Adam/learning_rate*
_output_shapes
: *
dtype0
}
dense_412/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *!
shared_namedense_412/kernel
v
$dense_412/kernel/Read/ReadVariableOpReadVariableOpdense_412/kernel*
_output_shapes
:	? *
dtype0
t
dense_412/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_412/bias
m
"dense_412/bias/Read/ReadVariableOpReadVariableOpdense_412/bias*
_output_shapes
: *
dtype0
}
dense_413/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*!
shared_namedense_413/kernel
v
$dense_413/kernel/Read/ReadVariableOpReadVariableOpdense_413/kernel*
_output_shapes
:	 ?*
dtype0
u
dense_413/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_413/bias
n
"dense_413/bias/Read/ReadVariableOpReadVariableOpdense_413/bias*
_output_shapes	
:?*
dtype0
~
dense_414/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_414/kernel
w
$dense_414/kernel/Read/ReadVariableOpReadVariableOpdense_414/kernel* 
_output_shapes
:
??*
dtype0
u
dense_414/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_414/bias
n
"dense_414/bias/Read/ReadVariableOpReadVariableOpdense_414/bias*
_output_shapes	
:?*
dtype0
~
dense_415/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_415/kernel
w
$dense_415/kernel/Read/ReadVariableOpReadVariableOpdense_415/kernel* 
_output_shapes
:
??*
dtype0
u
dense_415/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_415/bias
n
"dense_415/bias/Read/ReadVariableOpReadVariableOpdense_415/bias*
_output_shapes	
:?*
dtype0
~
dense_416/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_416/kernel
w
$dense_416/kernel/Read/ReadVariableOpReadVariableOpdense_416/kernel* 
_output_shapes
:
??*
dtype0
u
dense_416/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_416/bias
n
"dense_416/bias/Read/ReadVariableOpReadVariableOpdense_416/bias*
_output_shapes	
:?*
dtype0
~
dense_417/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_417/kernel
w
$dense_417/kernel/Read/ReadVariableOpReadVariableOpdense_417/kernel* 
_output_shapes
:
??*
dtype0
u
dense_417/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_417/bias
n
"dense_417/bias/Read/ReadVariableOpReadVariableOpdense_417/bias*
_output_shapes	
:?*
dtype0
~
dense_418/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_418/kernel
w
$dense_418/kernel/Read/ReadVariableOpReadVariableOpdense_418/kernel* 
_output_shapes
:
??*
dtype0
u
dense_418/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_418/bias
n
"dense_418/bias/Read/ReadVariableOpReadVariableOpdense_418/bias*
_output_shapes	
:?*
dtype0
}
dense_419/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_419/kernel
v
$dense_419/kernel/Read/ReadVariableOpReadVariableOpdense_419/kernel*
_output_shapes
:	?*
dtype0
t
dense_419/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_419/bias
m
"dense_419/bias/Read/ReadVariableOpReadVariableOpdense_419/bias*
_output_shapes
:*
dtype0
|
dense_420/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_420/kernel
u
$dense_420/kernel/Read/ReadVariableOpReadVariableOpdense_420/kernel*
_output_shapes

:*
dtype0
t
dense_420/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_420/bias
m
"dense_420/bias/Read/ReadVariableOpReadVariableOpdense_420/bias*
_output_shapes
:*
dtype0
?
training_130/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *'
shared_nametraining_130/Adam/iter
y
*training_130/Adam/iter/Read/ReadVariableOpReadVariableOptraining_130/Adam/iter*
_output_shapes
: *
dtype0	
?
training_130/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_130/Adam/beta_1
}
,training_130/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_130/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_130/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_130/Adam/beta_2
}
,training_130/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_130/Adam/beta_2*
_output_shapes
: *
dtype0
?
training_130/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametraining_130/Adam/decay
{
+training_130/Adam/decay/Read/ReadVariableOpReadVariableOptraining_130/Adam/decay*
_output_shapes
: *
dtype0
?
training_130/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training_130/Adam/learning_rate
?
3training_130/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_130/Adam/learning_rate*
_output_shapes
: *
dtype0
f
	total_329VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_329
_
total_329/Read/ReadVariableOpReadVariableOp	total_329*
_output_shapes
: *
dtype0
f
	count_329VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_329
_
count_329/Read/ReadVariableOpReadVariableOp	count_329*
_output_shapes
: *
dtype0
f
	total_139VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_139
_
total_139/Read/ReadVariableOpReadVariableOp	total_139*
_output_shapes
: *
dtype0
f
	count_139VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_139
_
count_139/Read/ReadVariableOpReadVariableOp	count_139*
_output_shapes
: *
dtype0
f
	total_140VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_140
_
total_140/Read/ReadVariableOpReadVariableOp	total_140*
_output_shapes
: *
dtype0
f
	count_140VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_140
_
count_140/Read/ReadVariableOpReadVariableOp	count_140*
_output_shapes
: *
dtype0
?
$training_318/Adam/dense_412/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *5
shared_name&$training_318/Adam/dense_412/kernel/m
?
8training_318/Adam/dense_412/kernel/m/Read/ReadVariableOpReadVariableOp$training_318/Adam/dense_412/kernel/m*
_output_shapes
:	? *
dtype0
?
"training_318/Adam/dense_412/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_318/Adam/dense_412/bias/m
?
6training_318/Adam/dense_412/bias/m/Read/ReadVariableOpReadVariableOp"training_318/Adam/dense_412/bias/m*
_output_shapes
: *
dtype0
?
$training_318/Adam/dense_413/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*5
shared_name&$training_318/Adam/dense_413/kernel/m
?
8training_318/Adam/dense_413/kernel/m/Read/ReadVariableOpReadVariableOp$training_318/Adam/dense_413/kernel/m*
_output_shapes
:	 ?*
dtype0
?
"training_318/Adam/dense_413/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_318/Adam/dense_413/bias/m
?
6training_318/Adam/dense_413/bias/m/Read/ReadVariableOpReadVariableOp"training_318/Adam/dense_413/bias/m*
_output_shapes	
:?*
dtype0
?
$training_318/Adam/dense_414/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_318/Adam/dense_414/kernel/m
?
8training_318/Adam/dense_414/kernel/m/Read/ReadVariableOpReadVariableOp$training_318/Adam/dense_414/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_318/Adam/dense_414/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_318/Adam/dense_414/bias/m
?
6training_318/Adam/dense_414/bias/m/Read/ReadVariableOpReadVariableOp"training_318/Adam/dense_414/bias/m*
_output_shapes	
:?*
dtype0
?
$training_318/Adam/dense_412/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *5
shared_name&$training_318/Adam/dense_412/kernel/v
?
8training_318/Adam/dense_412/kernel/v/Read/ReadVariableOpReadVariableOp$training_318/Adam/dense_412/kernel/v*
_output_shapes
:	? *
dtype0
?
"training_318/Adam/dense_412/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_318/Adam/dense_412/bias/v
?
6training_318/Adam/dense_412/bias/v/Read/ReadVariableOpReadVariableOp"training_318/Adam/dense_412/bias/v*
_output_shapes
: *
dtype0
?
$training_318/Adam/dense_413/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*5
shared_name&$training_318/Adam/dense_413/kernel/v
?
8training_318/Adam/dense_413/kernel/v/Read/ReadVariableOpReadVariableOp$training_318/Adam/dense_413/kernel/v*
_output_shapes
:	 ?*
dtype0
?
"training_318/Adam/dense_413/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_318/Adam/dense_413/bias/v
?
6training_318/Adam/dense_413/bias/v/Read/ReadVariableOpReadVariableOp"training_318/Adam/dense_413/bias/v*
_output_shapes	
:?*
dtype0
?
$training_318/Adam/dense_414/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_318/Adam/dense_414/kernel/v
?
8training_318/Adam/dense_414/kernel/v/Read/ReadVariableOpReadVariableOp$training_318/Adam/dense_414/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_318/Adam/dense_414/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_318/Adam/dense_414/bias/v
?
6training_318/Adam/dense_414/bias/v/Read/ReadVariableOpReadVariableOp"training_318/Adam/dense_414/bias/v*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_415/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_130/Adam/dense_415/kernel/m
?
8training_130/Adam/dense_415/kernel/m/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_415/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_130/Adam/dense_415/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_130/Adam/dense_415/bias/m
?
6training_130/Adam/dense_415/bias/m/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_415/bias/m*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_416/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_130/Adam/dense_416/kernel/m
?
8training_130/Adam/dense_416/kernel/m/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_416/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_130/Adam/dense_416/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_130/Adam/dense_416/bias/m
?
6training_130/Adam/dense_416/bias/m/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_416/bias/m*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_417/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_130/Adam/dense_417/kernel/m
?
8training_130/Adam/dense_417/kernel/m/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_417/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_130/Adam/dense_417/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_130/Adam/dense_417/bias/m
?
6training_130/Adam/dense_417/bias/m/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_417/bias/m*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_418/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_130/Adam/dense_418/kernel/m
?
8training_130/Adam/dense_418/kernel/m/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_418/kernel/m* 
_output_shapes
:
??*
dtype0
?
"training_130/Adam/dense_418/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_130/Adam/dense_418/bias/m
?
6training_130/Adam/dense_418/bias/m/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_418/bias/m*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_419/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*5
shared_name&$training_130/Adam/dense_419/kernel/m
?
8training_130/Adam/dense_419/kernel/m/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_419/kernel/m*
_output_shapes
:	?*
dtype0
?
"training_130/Adam/dense_419/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_130/Adam/dense_419/bias/m
?
6training_130/Adam/dense_419/bias/m/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_419/bias/m*
_output_shapes
:*
dtype0
?
$training_130/Adam/dense_420/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$training_130/Adam/dense_420/kernel/m
?
8training_130/Adam/dense_420/kernel/m/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_420/kernel/m*
_output_shapes

:*
dtype0
?
"training_130/Adam/dense_420/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_130/Adam/dense_420/bias/m
?
6training_130/Adam/dense_420/bias/m/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_420/bias/m*
_output_shapes
:*
dtype0
?
$training_130/Adam/dense_415/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_130/Adam/dense_415/kernel/v
?
8training_130/Adam/dense_415/kernel/v/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_415/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_130/Adam/dense_415/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_130/Adam/dense_415/bias/v
?
6training_130/Adam/dense_415/bias/v/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_415/bias/v*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_416/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_130/Adam/dense_416/kernel/v
?
8training_130/Adam/dense_416/kernel/v/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_416/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_130/Adam/dense_416/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_130/Adam/dense_416/bias/v
?
6training_130/Adam/dense_416/bias/v/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_416/bias/v*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_417/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_130/Adam/dense_417/kernel/v
?
8training_130/Adam/dense_417/kernel/v/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_417/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_130/Adam/dense_417/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_130/Adam/dense_417/bias/v
?
6training_130/Adam/dense_417/bias/v/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_417/bias/v*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_418/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$training_130/Adam/dense_418/kernel/v
?
8training_130/Adam/dense_418/kernel/v/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_418/kernel/v* 
_output_shapes
:
??*
dtype0
?
"training_130/Adam/dense_418/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"training_130/Adam/dense_418/bias/v
?
6training_130/Adam/dense_418/bias/v/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_418/bias/v*
_output_shapes	
:?*
dtype0
?
$training_130/Adam/dense_419/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*5
shared_name&$training_130/Adam/dense_419/kernel/v
?
8training_130/Adam/dense_419/kernel/v/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_419/kernel/v*
_output_shapes
:	?*
dtype0
?
"training_130/Adam/dense_419/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_130/Adam/dense_419/bias/v
?
6training_130/Adam/dense_419/bias/v/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_419/bias/v*
_output_shapes
:*
dtype0
?
$training_130/Adam/dense_420/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$training_130/Adam/dense_420/kernel/v
?
8training_130/Adam/dense_420/kernel/v/Read/ReadVariableOpReadVariableOp$training_130/Adam/dense_420/kernel/v*
_output_shapes

:*
dtype0
?
"training_130/Adam/dense_420/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_130/Adam/dense_420/bias/v
?
6training_130/Adam/dense_420/bias/v/Read/ReadVariableOpReadVariableOp"training_130/Adam/dense_420/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?s
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?s
value?rB?r B?r
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
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
layer_with_weights-5
layer-8
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
415
516
617
?
regularization_losses
trainable_variables

7layers
8layer_regularization_losses
9metrics
:layer_metrics
;non_trainable_variables
	variables
 
h

%kernel
&bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
h

'kernel
(bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
h

)kernel
*bias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
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
regularization_losses
trainable_variables

Hlayers
Ilayer_regularization_losses
Jmetrics
Klayer_metrics
Lnon_trainable_variables
	variables
 
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
h

+kernel
,bias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
h

-kernel
.bias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
h

/kernel
0bias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
h

1kernel
2bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
h

3kernel
4bias
aregularization_losses
btrainable_variables
c	variables
d	keras_api
R
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
h

5kernel
6bias
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
?
miter

nbeta_1

obeta_2
	pdecay
qlearning_rate+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?
 
 
V
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
510
611
?
regularization_losses
trainable_variables

rlayers
slayer_regularization_losses
tmetrics
ulayer_metrics
vnon_trainable_variables
	variables
US
VARIABLE_VALUEtraining_318/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEtraining_318/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEtraining_318/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_318/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEtraining_318/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_412/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_412/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_413/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_413/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_414/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_414/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_415/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_415/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_416/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_416/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_417/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_417/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_418/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_418/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_419/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_419/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdense_420/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_420/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

w0
 
V
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
510
611
 

%0
&1

%0
&1
?
<regularization_losses
=trainable_variables
xlayer_regularization_losses

ylayers
zmetrics
{layer_metrics
|non_trainable_variables
>	variables
 

'0
(1

'0
(1
?
@regularization_losses
Atrainable_variables
}layer_regularization_losses

~layers
metrics
?layer_metrics
?non_trainable_variables
B	variables
 

)0
*1

)0
*1
?
Dregularization_losses
Etrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
F	variables

0

1
2
3
 

?0
 
 
 
 
 
?
Mregularization_losses
Ntrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
O	variables
 
 

+0
,1
?
Qregularization_losses
Rtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
S	variables
 
 

-0
.1
?
Uregularization_losses
Vtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
W	variables
 
 

/0
01
?
Yregularization_losses
Ztrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
[	variables
 
 

10
21
?
]regularization_losses
^trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
_	variables
 
 

30
41
?
aregularization_losses
btrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
c	variables
 
 
 
?
eregularization_losses
ftrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
g	variables
 
 

50
61
?
iregularization_losses
jtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
k	variables
jh
VARIABLE_VALUEtraining_130/Adam/iter>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEtraining_130/Adam/beta_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEtraining_130/Adam/beta_2@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEtraining_130/Adam/decay?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEtraining_130/Adam/learning_rateGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
8
 

?0
 
V
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
510
611
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

30
41
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
50
61
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
SQ
VARIABLE_VALUE	total_3294keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	count_3294keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
hf
VARIABLE_VALUE	total_139Ilayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE	count_139Ilayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
hf
VARIABLE_VALUE	total_140Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE	count_140Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE$training_318/Adam/dense_412/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_318/Adam/dense_412/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_318/Adam/dense_413/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_318/Adam/dense_413/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_318/Adam/dense_414/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_318/Adam/dense_414/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_318/Adam/dense_412/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_318/Adam/dense_412/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_318/Adam/dense_413/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_318/Adam/dense_413/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_318/Adam/dense_414/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_318/Adam/dense_414/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_415/kernel/mWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_415/bias/mWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_416/kernel/mWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_416/bias/mWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_417/kernel/mXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_417/bias/mXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_418/kernel/mXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_418/bias/mXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_419/kernel/mXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_419/bias/mXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_420/kernel/mXvariables/16/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_420/bias/mXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_415/kernel/vWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_415/bias/vWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_416/kernel/vWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_416/bias/vWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_417/kernel/vXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_417/bias/vXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_418/kernel/vXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_418/bias/vXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_419/kernel/vXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_419/bias/vXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$training_130/Adam/dense_420/kernel/vXvariables/16/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_130/Adam/dense_420/bias/vXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_11Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11dense_412/kerneldense_412/biasdense_413/kerneldense_413/biasdense_414/kerneldense_414/biasdense_415/kerneldense_415/biasdense_416/kerneldense_416/biasdense_417/kerneldense_417/biasdense_418/kerneldense_418/biasdense_419/kerneldense_419/biasdense_420/kerneldense_420/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_60989254
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*training_318/Adam/iter/Read/ReadVariableOp,training_318/Adam/beta_1/Read/ReadVariableOp,training_318/Adam/beta_2/Read/ReadVariableOp+training_318/Adam/decay/Read/ReadVariableOp3training_318/Adam/learning_rate/Read/ReadVariableOp$dense_412/kernel/Read/ReadVariableOp"dense_412/bias/Read/ReadVariableOp$dense_413/kernel/Read/ReadVariableOp"dense_413/bias/Read/ReadVariableOp$dense_414/kernel/Read/ReadVariableOp"dense_414/bias/Read/ReadVariableOp$dense_415/kernel/Read/ReadVariableOp"dense_415/bias/Read/ReadVariableOp$dense_416/kernel/Read/ReadVariableOp"dense_416/bias/Read/ReadVariableOp$dense_417/kernel/Read/ReadVariableOp"dense_417/bias/Read/ReadVariableOp$dense_418/kernel/Read/ReadVariableOp"dense_418/bias/Read/ReadVariableOp$dense_419/kernel/Read/ReadVariableOp"dense_419/bias/Read/ReadVariableOp$dense_420/kernel/Read/ReadVariableOp"dense_420/bias/Read/ReadVariableOp*training_130/Adam/iter/Read/ReadVariableOp,training_130/Adam/beta_1/Read/ReadVariableOp,training_130/Adam/beta_2/Read/ReadVariableOp+training_130/Adam/decay/Read/ReadVariableOp3training_130/Adam/learning_rate/Read/ReadVariableOptotal_329/Read/ReadVariableOpcount_329/Read/ReadVariableOptotal_139/Read/ReadVariableOpcount_139/Read/ReadVariableOptotal_140/Read/ReadVariableOpcount_140/Read/ReadVariableOp8training_318/Adam/dense_412/kernel/m/Read/ReadVariableOp6training_318/Adam/dense_412/bias/m/Read/ReadVariableOp8training_318/Adam/dense_413/kernel/m/Read/ReadVariableOp6training_318/Adam/dense_413/bias/m/Read/ReadVariableOp8training_318/Adam/dense_414/kernel/m/Read/ReadVariableOp6training_318/Adam/dense_414/bias/m/Read/ReadVariableOp8training_318/Adam/dense_412/kernel/v/Read/ReadVariableOp6training_318/Adam/dense_412/bias/v/Read/ReadVariableOp8training_318/Adam/dense_413/kernel/v/Read/ReadVariableOp6training_318/Adam/dense_413/bias/v/Read/ReadVariableOp8training_318/Adam/dense_414/kernel/v/Read/ReadVariableOp6training_318/Adam/dense_414/bias/v/Read/ReadVariableOp8training_130/Adam/dense_415/kernel/m/Read/ReadVariableOp6training_130/Adam/dense_415/bias/m/Read/ReadVariableOp8training_130/Adam/dense_416/kernel/m/Read/ReadVariableOp6training_130/Adam/dense_416/bias/m/Read/ReadVariableOp8training_130/Adam/dense_417/kernel/m/Read/ReadVariableOp6training_130/Adam/dense_417/bias/m/Read/ReadVariableOp8training_130/Adam/dense_418/kernel/m/Read/ReadVariableOp6training_130/Adam/dense_418/bias/m/Read/ReadVariableOp8training_130/Adam/dense_419/kernel/m/Read/ReadVariableOp6training_130/Adam/dense_419/bias/m/Read/ReadVariableOp8training_130/Adam/dense_420/kernel/m/Read/ReadVariableOp6training_130/Adam/dense_420/bias/m/Read/ReadVariableOp8training_130/Adam/dense_415/kernel/v/Read/ReadVariableOp6training_130/Adam/dense_415/bias/v/Read/ReadVariableOp8training_130/Adam/dense_416/kernel/v/Read/ReadVariableOp6training_130/Adam/dense_416/bias/v/Read/ReadVariableOp8training_130/Adam/dense_417/kernel/v/Read/ReadVariableOp6training_130/Adam/dense_417/bias/v/Read/ReadVariableOp8training_130/Adam/dense_418/kernel/v/Read/ReadVariableOp6training_130/Adam/dense_418/bias/v/Read/ReadVariableOp8training_130/Adam/dense_419/kernel/v/Read/ReadVariableOp6training_130/Adam/dense_419/bias/v/Read/ReadVariableOp8training_130/Adam/dense_420/kernel/v/Read/ReadVariableOp6training_130/Adam/dense_420/bias/v/Read/ReadVariableOpConst*S
TinL
J2H		*
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
!__inference__traced_save_60991076
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametraining_318/Adam/itertraining_318/Adam/beta_1training_318/Adam/beta_2training_318/Adam/decaytraining_318/Adam/learning_ratedense_412/kerneldense_412/biasdense_413/kerneldense_413/biasdense_414/kerneldense_414/biasdense_415/kerneldense_415/biasdense_416/kerneldense_416/biasdense_417/kerneldense_417/biasdense_418/kerneldense_418/biasdense_419/kerneldense_419/biasdense_420/kerneldense_420/biastraining_130/Adam/itertraining_130/Adam/beta_1training_130/Adam/beta_2training_130/Adam/decaytraining_130/Adam/learning_rate	total_329	count_329	total_139	count_139	total_140	count_140$training_318/Adam/dense_412/kernel/m"training_318/Adam/dense_412/bias/m$training_318/Adam/dense_413/kernel/m"training_318/Adam/dense_413/bias/m$training_318/Adam/dense_414/kernel/m"training_318/Adam/dense_414/bias/m$training_318/Adam/dense_412/kernel/v"training_318/Adam/dense_412/bias/v$training_318/Adam/dense_413/kernel/v"training_318/Adam/dense_413/bias/v$training_318/Adam/dense_414/kernel/v"training_318/Adam/dense_414/bias/v$training_130/Adam/dense_415/kernel/m"training_130/Adam/dense_415/bias/m$training_130/Adam/dense_416/kernel/m"training_130/Adam/dense_416/bias/m$training_130/Adam/dense_417/kernel/m"training_130/Adam/dense_417/bias/m$training_130/Adam/dense_418/kernel/m"training_130/Adam/dense_418/bias/m$training_130/Adam/dense_419/kernel/m"training_130/Adam/dense_419/bias/m$training_130/Adam/dense_420/kernel/m"training_130/Adam/dense_420/bias/m$training_130/Adam/dense_415/kernel/v"training_130/Adam/dense_415/bias/v$training_130/Adam/dense_416/kernel/v"training_130/Adam/dense_416/bias/v$training_130/Adam/dense_417/kernel/v"training_130/Adam/dense_417/bias/v$training_130/Adam/dense_418/kernel/v"training_130/Adam/dense_418/bias/v$training_130/Adam/dense_419/kernel/v"training_130/Adam/dense_419/bias/v$training_130/Adam/dense_420/kernel/v"training_130/Adam/dense_420/bias/v*R
TinK
I2G*
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
$__inference__traced_restore_60991296??
??
?
G__inference_model_960_layer_call_and_return_conditional_losses_60989719

inputsC
?autoencoder_dense_412_tensordot_readvariableop_dense_412_kernel?
;autoencoder_dense_412_biasadd_readvariableop_dense_412_biasC
?autoencoder_dense_413_tensordot_readvariableop_dense_413_kernel?
;autoencoder_dense_413_biasadd_readvariableop_dense_413_biasC
?autoencoder_dense_414_tensordot_readvariableop_dense_414_kernel?
;autoencoder_dense_414_biasadd_readvariableop_dense_414_biasE
Adiscriminator_dense_415_tensordot_readvariableop_dense_415_kernelA
=discriminator_dense_415_biasadd_readvariableop_dense_415_biasE
Adiscriminator_dense_416_tensordot_readvariableop_dense_416_kernelA
=discriminator_dense_416_biasadd_readvariableop_dense_416_biasE
Adiscriminator_dense_417_tensordot_readvariableop_dense_417_kernelA
=discriminator_dense_417_biasadd_readvariableop_dense_417_biasE
Adiscriminator_dense_418_tensordot_readvariableop_dense_418_kernelA
=discriminator_dense_418_biasadd_readvariableop_dense_418_biasE
Adiscriminator_dense_419_tensordot_readvariableop_dense_419_kernelA
=discriminator_dense_419_biasadd_readvariableop_dense_419_biasB
>discriminator_dense_420_matmul_readvariableop_dense_420_kernelA
=discriminator_dense_420_biasadd_readvariableop_dense_420_bias
identity??,autoencoder/dense_412/BiasAdd/ReadVariableOp?.autoencoder/dense_412/Tensordot/ReadVariableOp?,autoencoder/dense_413/BiasAdd/ReadVariableOp?.autoencoder/dense_413/Tensordot/ReadVariableOp?,autoencoder/dense_414/BiasAdd/ReadVariableOp?.autoencoder/dense_414/Tensordot/ReadVariableOp?.discriminator/dense_415/BiasAdd/ReadVariableOp?0discriminator/dense_415/Tensordot/ReadVariableOp?.discriminator/dense_416/BiasAdd/ReadVariableOp?0discriminator/dense_416/Tensordot/ReadVariableOp?.discriminator/dense_417/BiasAdd/ReadVariableOp?0discriminator/dense_417/Tensordot/ReadVariableOp?.discriminator/dense_418/BiasAdd/ReadVariableOp?0discriminator/dense_418/Tensordot/ReadVariableOp?.discriminator/dense_419/BiasAdd/ReadVariableOp?0discriminator/dense_419/Tensordot/ReadVariableOp?.discriminator/dense_420/BiasAdd/ReadVariableOp?-discriminator/dense_420/MatMul/ReadVariableOp?
.autoencoder/dense_412/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype020
.autoencoder/dense_412/Tensordot/ReadVariableOp?
$autoencoder/dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_412/Tensordot/axes?
$autoencoder/dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_412/Tensordot/free?
%autoencoder/dense_412/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2'
%autoencoder/dense_412/Tensordot/Shape?
-autoencoder/dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_412/Tensordot/GatherV2/axis?
(autoencoder/dense_412/Tensordot/GatherV2GatherV2.autoencoder/dense_412/Tensordot/Shape:output:0-autoencoder/dense_412/Tensordot/free:output:06autoencoder/dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_412/Tensordot/GatherV2?
/autoencoder/dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_412/Tensordot/GatherV2_1/axis?
*autoencoder/dense_412/Tensordot/GatherV2_1GatherV2.autoencoder/dense_412/Tensordot/Shape:output:0-autoencoder/dense_412/Tensordot/axes:output:08autoencoder/dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_412/Tensordot/GatherV2_1?
%autoencoder/dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_412/Tensordot/Const?
$autoencoder/dense_412/Tensordot/ProdProd1autoencoder/dense_412/Tensordot/GatherV2:output:0.autoencoder/dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_412/Tensordot/Prod?
'autoencoder/dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_412/Tensordot/Const_1?
&autoencoder/dense_412/Tensordot/Prod_1Prod3autoencoder/dense_412/Tensordot/GatherV2_1:output:00autoencoder/dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_412/Tensordot/Prod_1?
+autoencoder/dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_412/Tensordot/concat/axis?
&autoencoder/dense_412/Tensordot/concatConcatV2-autoencoder/dense_412/Tensordot/free:output:0-autoencoder/dense_412/Tensordot/axes:output:04autoencoder/dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_412/Tensordot/concat?
%autoencoder/dense_412/Tensordot/stackPack-autoencoder/dense_412/Tensordot/Prod:output:0/autoencoder/dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_412/Tensordot/stack?
)autoencoder/dense_412/Tensordot/transpose	Transposeinputs/autoencoder/dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)autoencoder/dense_412/Tensordot/transpose?
'autoencoder/dense_412/Tensordot/ReshapeReshape-autoencoder/dense_412/Tensordot/transpose:y:0.autoencoder/dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_412/Tensordot/Reshape?
&autoencoder/dense_412/Tensordot/MatMulMatMul0autoencoder/dense_412/Tensordot/Reshape:output:06autoencoder/dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&autoencoder/dense_412/Tensordot/MatMul?
'autoencoder/dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_412/Tensordot/Const_2?
-autoencoder/dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_412/Tensordot/concat_1/axis?
(autoencoder/dense_412/Tensordot/concat_1ConcatV21autoencoder/dense_412/Tensordot/GatherV2:output:00autoencoder/dense_412/Tensordot/Const_2:output:06autoencoder/dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_412/Tensordot/concat_1?
autoencoder/dense_412/TensordotReshape0autoencoder/dense_412/Tensordot/MatMul:product:01autoencoder/dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2!
autoencoder/dense_412/Tensordot?
,autoencoder/dense_412/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02.
,autoencoder/dense_412/BiasAdd/ReadVariableOp?
autoencoder/dense_412/BiasAddBiasAdd(autoencoder/dense_412/Tensordot:output:04autoencoder/dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
autoencoder/dense_412/BiasAdd?
autoencoder/dense_412/TanhTanh&autoencoder/dense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
autoencoder/dense_412/Tanh?
.autoencoder/dense_413/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype020
.autoencoder/dense_413/Tensordot/ReadVariableOp?
$autoencoder/dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_413/Tensordot/axes?
$autoencoder/dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_413/Tensordot/free?
%autoencoder/dense_413/Tensordot/ShapeShapeautoencoder/dense_412/Tanh:y:0*
T0*
_output_shapes
:2'
%autoencoder/dense_413/Tensordot/Shape?
-autoencoder/dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_413/Tensordot/GatherV2/axis?
(autoencoder/dense_413/Tensordot/GatherV2GatherV2.autoencoder/dense_413/Tensordot/Shape:output:0-autoencoder/dense_413/Tensordot/free:output:06autoencoder/dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_413/Tensordot/GatherV2?
/autoencoder/dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_413/Tensordot/GatherV2_1/axis?
*autoencoder/dense_413/Tensordot/GatherV2_1GatherV2.autoencoder/dense_413/Tensordot/Shape:output:0-autoencoder/dense_413/Tensordot/axes:output:08autoencoder/dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_413/Tensordot/GatherV2_1?
%autoencoder/dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_413/Tensordot/Const?
$autoencoder/dense_413/Tensordot/ProdProd1autoencoder/dense_413/Tensordot/GatherV2:output:0.autoencoder/dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_413/Tensordot/Prod?
'autoencoder/dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_413/Tensordot/Const_1?
&autoencoder/dense_413/Tensordot/Prod_1Prod3autoencoder/dense_413/Tensordot/GatherV2_1:output:00autoencoder/dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_413/Tensordot/Prod_1?
+autoencoder/dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_413/Tensordot/concat/axis?
&autoencoder/dense_413/Tensordot/concatConcatV2-autoencoder/dense_413/Tensordot/free:output:0-autoencoder/dense_413/Tensordot/axes:output:04autoencoder/dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_413/Tensordot/concat?
%autoencoder/dense_413/Tensordot/stackPack-autoencoder/dense_413/Tensordot/Prod:output:0/autoencoder/dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_413/Tensordot/stack?
)autoencoder/dense_413/Tensordot/transpose	Transposeautoencoder/dense_412/Tanh:y:0/autoencoder/dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2+
)autoencoder/dense_413/Tensordot/transpose?
'autoencoder/dense_413/Tensordot/ReshapeReshape-autoencoder/dense_413/Tensordot/transpose:y:0.autoencoder/dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_413/Tensordot/Reshape?
&autoencoder/dense_413/Tensordot/MatMulMatMul0autoencoder/dense_413/Tensordot/Reshape:output:06autoencoder/dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/dense_413/Tensordot/MatMul?
'autoencoder/dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'autoencoder/dense_413/Tensordot/Const_2?
-autoencoder/dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_413/Tensordot/concat_1/axis?
(autoencoder/dense_413/Tensordot/concat_1ConcatV21autoencoder/dense_413/Tensordot/GatherV2:output:00autoencoder/dense_413/Tensordot/Const_2:output:06autoencoder/dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_413/Tensordot/concat_1?
autoencoder/dense_413/TensordotReshape0autoencoder/dense_413/Tensordot/MatMul:product:01autoencoder/dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
autoencoder/dense_413/Tensordot?
,autoencoder/dense_413/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02.
,autoencoder/dense_413/BiasAdd/ReadVariableOp?
autoencoder/dense_413/BiasAddBiasAdd(autoencoder/dense_413/Tensordot:output:04autoencoder/dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_413/BiasAdd?
autoencoder/dense_413/TanhTanh&autoencoder/dense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_413/Tanh?
.autoencoder/dense_414/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype020
.autoencoder/dense_414/Tensordot/ReadVariableOp?
$autoencoder/dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_414/Tensordot/axes?
$autoencoder/dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_414/Tensordot/free?
%autoencoder/dense_414/Tensordot/ShapeShapeautoencoder/dense_413/Tanh:y:0*
T0*
_output_shapes
:2'
%autoencoder/dense_414/Tensordot/Shape?
-autoencoder/dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_414/Tensordot/GatherV2/axis?
(autoencoder/dense_414/Tensordot/GatherV2GatherV2.autoencoder/dense_414/Tensordot/Shape:output:0-autoencoder/dense_414/Tensordot/free:output:06autoencoder/dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_414/Tensordot/GatherV2?
/autoencoder/dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_414/Tensordot/GatherV2_1/axis?
*autoencoder/dense_414/Tensordot/GatherV2_1GatherV2.autoencoder/dense_414/Tensordot/Shape:output:0-autoencoder/dense_414/Tensordot/axes:output:08autoencoder/dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_414/Tensordot/GatherV2_1?
%autoencoder/dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_414/Tensordot/Const?
$autoencoder/dense_414/Tensordot/ProdProd1autoencoder/dense_414/Tensordot/GatherV2:output:0.autoencoder/dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_414/Tensordot/Prod?
'autoencoder/dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_414/Tensordot/Const_1?
&autoencoder/dense_414/Tensordot/Prod_1Prod3autoencoder/dense_414/Tensordot/GatherV2_1:output:00autoencoder/dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_414/Tensordot/Prod_1?
+autoencoder/dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_414/Tensordot/concat/axis?
&autoencoder/dense_414/Tensordot/concatConcatV2-autoencoder/dense_414/Tensordot/free:output:0-autoencoder/dense_414/Tensordot/axes:output:04autoencoder/dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_414/Tensordot/concat?
%autoencoder/dense_414/Tensordot/stackPack-autoencoder/dense_414/Tensordot/Prod:output:0/autoencoder/dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_414/Tensordot/stack?
)autoencoder/dense_414/Tensordot/transpose	Transposeautoencoder/dense_413/Tanh:y:0/autoencoder/dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)autoencoder/dense_414/Tensordot/transpose?
'autoencoder/dense_414/Tensordot/ReshapeReshape-autoencoder/dense_414/Tensordot/transpose:y:0.autoencoder/dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_414/Tensordot/Reshape?
&autoencoder/dense_414/Tensordot/MatMulMatMul0autoencoder/dense_414/Tensordot/Reshape:output:06autoencoder/dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/dense_414/Tensordot/MatMul?
'autoencoder/dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'autoencoder/dense_414/Tensordot/Const_2?
-autoencoder/dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_414/Tensordot/concat_1/axis?
(autoencoder/dense_414/Tensordot/concat_1ConcatV21autoencoder/dense_414/Tensordot/GatherV2:output:00autoencoder/dense_414/Tensordot/Const_2:output:06autoencoder/dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_414/Tensordot/concat_1?
autoencoder/dense_414/TensordotReshape0autoencoder/dense_414/Tensordot/MatMul:product:01autoencoder/dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
autoencoder/dense_414/Tensordot?
,autoencoder/dense_414/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02.
,autoencoder/dense_414/BiasAdd/ReadVariableOp?
autoencoder/dense_414/BiasAddBiasAdd(autoencoder/dense_414/Tensordot:output:04autoencoder/dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_414/BiasAdd?
discriminator/dropout/IdentityIdentity&autoencoder/dense_414/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2 
discriminator/dropout/Identity?
0discriminator/dense_415/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_415_tensordot_readvariableop_dense_415_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_415/Tensordot/ReadVariableOp?
&discriminator/dense_415/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_415/Tensordot/axes?
&discriminator/dense_415/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_415/Tensordot/free?
'discriminator/dense_415/Tensordot/ShapeShape'discriminator/dropout/Identity:output:0*
T0*
_output_shapes
:2)
'discriminator/dense_415/Tensordot/Shape?
/discriminator/dense_415/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_415/Tensordot/GatherV2/axis?
*discriminator/dense_415/Tensordot/GatherV2GatherV20discriminator/dense_415/Tensordot/Shape:output:0/discriminator/dense_415/Tensordot/free:output:08discriminator/dense_415/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_415/Tensordot/GatherV2?
1discriminator/dense_415/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_415/Tensordot/GatherV2_1/axis?
,discriminator/dense_415/Tensordot/GatherV2_1GatherV20discriminator/dense_415/Tensordot/Shape:output:0/discriminator/dense_415/Tensordot/axes:output:0:discriminator/dense_415/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_415/Tensordot/GatherV2_1?
'discriminator/dense_415/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_415/Tensordot/Const?
&discriminator/dense_415/Tensordot/ProdProd3discriminator/dense_415/Tensordot/GatherV2:output:00discriminator/dense_415/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_415/Tensordot/Prod?
)discriminator/dense_415/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_415/Tensordot/Const_1?
(discriminator/dense_415/Tensordot/Prod_1Prod5discriminator/dense_415/Tensordot/GatherV2_1:output:02discriminator/dense_415/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_415/Tensordot/Prod_1?
-discriminator/dense_415/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_415/Tensordot/concat/axis?
(discriminator/dense_415/Tensordot/concatConcatV2/discriminator/dense_415/Tensordot/free:output:0/discriminator/dense_415/Tensordot/axes:output:06discriminator/dense_415/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_415/Tensordot/concat?
'discriminator/dense_415/Tensordot/stackPack/discriminator/dense_415/Tensordot/Prod:output:01discriminator/dense_415/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_415/Tensordot/stack?
+discriminator/dense_415/Tensordot/transpose	Transpose'discriminator/dropout/Identity:output:01discriminator/dense_415/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_415/Tensordot/transpose?
)discriminator/dense_415/Tensordot/ReshapeReshape/discriminator/dense_415/Tensordot/transpose:y:00discriminator/dense_415/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_415/Tensordot/Reshape?
(discriminator/dense_415/Tensordot/MatMulMatMul2discriminator/dense_415/Tensordot/Reshape:output:08discriminator/dense_415/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_415/Tensordot/MatMul?
)discriminator/dense_415/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_415/Tensordot/Const_2?
/discriminator/dense_415/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_415/Tensordot/concat_1/axis?
*discriminator/dense_415/Tensordot/concat_1ConcatV23discriminator/dense_415/Tensordot/GatherV2:output:02discriminator/dense_415/Tensordot/Const_2:output:08discriminator/dense_415/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_415/Tensordot/concat_1?
!discriminator/dense_415/TensordotReshape2discriminator/dense_415/Tensordot/MatMul:product:03discriminator/dense_415/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_415/Tensordot?
.discriminator/dense_415/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_415_biasadd_readvariableop_dense_415_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_415/BiasAdd/ReadVariableOp?
discriminator/dense_415/BiasAddBiasAdd*discriminator/dense_415/Tensordot:output:06discriminator/dense_415/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_415/BiasAdd?
discriminator/dense_415/TanhTanh(discriminator/dense_415/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_415/Tanh?
0discriminator/dense_416/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_416_tensordot_readvariableop_dense_416_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_416/Tensordot/ReadVariableOp?
&discriminator/dense_416/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_416/Tensordot/axes?
&discriminator/dense_416/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_416/Tensordot/free?
'discriminator/dense_416/Tensordot/ShapeShape discriminator/dense_415/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_416/Tensordot/Shape?
/discriminator/dense_416/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_416/Tensordot/GatherV2/axis?
*discriminator/dense_416/Tensordot/GatherV2GatherV20discriminator/dense_416/Tensordot/Shape:output:0/discriminator/dense_416/Tensordot/free:output:08discriminator/dense_416/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_416/Tensordot/GatherV2?
1discriminator/dense_416/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_416/Tensordot/GatherV2_1/axis?
,discriminator/dense_416/Tensordot/GatherV2_1GatherV20discriminator/dense_416/Tensordot/Shape:output:0/discriminator/dense_416/Tensordot/axes:output:0:discriminator/dense_416/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_416/Tensordot/GatherV2_1?
'discriminator/dense_416/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_416/Tensordot/Const?
&discriminator/dense_416/Tensordot/ProdProd3discriminator/dense_416/Tensordot/GatherV2:output:00discriminator/dense_416/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_416/Tensordot/Prod?
)discriminator/dense_416/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_416/Tensordot/Const_1?
(discriminator/dense_416/Tensordot/Prod_1Prod5discriminator/dense_416/Tensordot/GatherV2_1:output:02discriminator/dense_416/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_416/Tensordot/Prod_1?
-discriminator/dense_416/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_416/Tensordot/concat/axis?
(discriminator/dense_416/Tensordot/concatConcatV2/discriminator/dense_416/Tensordot/free:output:0/discriminator/dense_416/Tensordot/axes:output:06discriminator/dense_416/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_416/Tensordot/concat?
'discriminator/dense_416/Tensordot/stackPack/discriminator/dense_416/Tensordot/Prod:output:01discriminator/dense_416/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_416/Tensordot/stack?
+discriminator/dense_416/Tensordot/transpose	Transpose discriminator/dense_415/Tanh:y:01discriminator/dense_416/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_416/Tensordot/transpose?
)discriminator/dense_416/Tensordot/ReshapeReshape/discriminator/dense_416/Tensordot/transpose:y:00discriminator/dense_416/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_416/Tensordot/Reshape?
(discriminator/dense_416/Tensordot/MatMulMatMul2discriminator/dense_416/Tensordot/Reshape:output:08discriminator/dense_416/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_416/Tensordot/MatMul?
)discriminator/dense_416/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_416/Tensordot/Const_2?
/discriminator/dense_416/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_416/Tensordot/concat_1/axis?
*discriminator/dense_416/Tensordot/concat_1ConcatV23discriminator/dense_416/Tensordot/GatherV2:output:02discriminator/dense_416/Tensordot/Const_2:output:08discriminator/dense_416/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_416/Tensordot/concat_1?
!discriminator/dense_416/TensordotReshape2discriminator/dense_416/Tensordot/MatMul:product:03discriminator/dense_416/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_416/Tensordot?
.discriminator/dense_416/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_416_biasadd_readvariableop_dense_416_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_416/BiasAdd/ReadVariableOp?
discriminator/dense_416/BiasAddBiasAdd*discriminator/dense_416/Tensordot:output:06discriminator/dense_416/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_416/BiasAdd?
discriminator/dense_416/TanhTanh(discriminator/dense_416/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_416/Tanh?
0discriminator/dense_417/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_417_tensordot_readvariableop_dense_417_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_417/Tensordot/ReadVariableOp?
&discriminator/dense_417/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_417/Tensordot/axes?
&discriminator/dense_417/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_417/Tensordot/free?
'discriminator/dense_417/Tensordot/ShapeShape discriminator/dense_416/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_417/Tensordot/Shape?
/discriminator/dense_417/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_417/Tensordot/GatherV2/axis?
*discriminator/dense_417/Tensordot/GatherV2GatherV20discriminator/dense_417/Tensordot/Shape:output:0/discriminator/dense_417/Tensordot/free:output:08discriminator/dense_417/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_417/Tensordot/GatherV2?
1discriminator/dense_417/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_417/Tensordot/GatherV2_1/axis?
,discriminator/dense_417/Tensordot/GatherV2_1GatherV20discriminator/dense_417/Tensordot/Shape:output:0/discriminator/dense_417/Tensordot/axes:output:0:discriminator/dense_417/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_417/Tensordot/GatherV2_1?
'discriminator/dense_417/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_417/Tensordot/Const?
&discriminator/dense_417/Tensordot/ProdProd3discriminator/dense_417/Tensordot/GatherV2:output:00discriminator/dense_417/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_417/Tensordot/Prod?
)discriminator/dense_417/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_417/Tensordot/Const_1?
(discriminator/dense_417/Tensordot/Prod_1Prod5discriminator/dense_417/Tensordot/GatherV2_1:output:02discriminator/dense_417/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_417/Tensordot/Prod_1?
-discriminator/dense_417/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_417/Tensordot/concat/axis?
(discriminator/dense_417/Tensordot/concatConcatV2/discriminator/dense_417/Tensordot/free:output:0/discriminator/dense_417/Tensordot/axes:output:06discriminator/dense_417/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_417/Tensordot/concat?
'discriminator/dense_417/Tensordot/stackPack/discriminator/dense_417/Tensordot/Prod:output:01discriminator/dense_417/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_417/Tensordot/stack?
+discriminator/dense_417/Tensordot/transpose	Transpose discriminator/dense_416/Tanh:y:01discriminator/dense_417/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_417/Tensordot/transpose?
)discriminator/dense_417/Tensordot/ReshapeReshape/discriminator/dense_417/Tensordot/transpose:y:00discriminator/dense_417/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_417/Tensordot/Reshape?
(discriminator/dense_417/Tensordot/MatMulMatMul2discriminator/dense_417/Tensordot/Reshape:output:08discriminator/dense_417/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_417/Tensordot/MatMul?
)discriminator/dense_417/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_417/Tensordot/Const_2?
/discriminator/dense_417/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_417/Tensordot/concat_1/axis?
*discriminator/dense_417/Tensordot/concat_1ConcatV23discriminator/dense_417/Tensordot/GatherV2:output:02discriminator/dense_417/Tensordot/Const_2:output:08discriminator/dense_417/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_417/Tensordot/concat_1?
!discriminator/dense_417/TensordotReshape2discriminator/dense_417/Tensordot/MatMul:product:03discriminator/dense_417/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_417/Tensordot?
.discriminator/dense_417/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_417_biasadd_readvariableop_dense_417_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_417/BiasAdd/ReadVariableOp?
discriminator/dense_417/BiasAddBiasAdd*discriminator/dense_417/Tensordot:output:06discriminator/dense_417/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_417/BiasAdd?
discriminator/dense_417/TanhTanh(discriminator/dense_417/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_417/Tanh?
0discriminator/dense_418/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_418_tensordot_readvariableop_dense_418_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_418/Tensordot/ReadVariableOp?
&discriminator/dense_418/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_418/Tensordot/axes?
&discriminator/dense_418/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_418/Tensordot/free?
'discriminator/dense_418/Tensordot/ShapeShape discriminator/dense_417/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_418/Tensordot/Shape?
/discriminator/dense_418/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_418/Tensordot/GatherV2/axis?
*discriminator/dense_418/Tensordot/GatherV2GatherV20discriminator/dense_418/Tensordot/Shape:output:0/discriminator/dense_418/Tensordot/free:output:08discriminator/dense_418/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_418/Tensordot/GatherV2?
1discriminator/dense_418/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_418/Tensordot/GatherV2_1/axis?
,discriminator/dense_418/Tensordot/GatherV2_1GatherV20discriminator/dense_418/Tensordot/Shape:output:0/discriminator/dense_418/Tensordot/axes:output:0:discriminator/dense_418/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_418/Tensordot/GatherV2_1?
'discriminator/dense_418/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_418/Tensordot/Const?
&discriminator/dense_418/Tensordot/ProdProd3discriminator/dense_418/Tensordot/GatherV2:output:00discriminator/dense_418/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_418/Tensordot/Prod?
)discriminator/dense_418/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_418/Tensordot/Const_1?
(discriminator/dense_418/Tensordot/Prod_1Prod5discriminator/dense_418/Tensordot/GatherV2_1:output:02discriminator/dense_418/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_418/Tensordot/Prod_1?
-discriminator/dense_418/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_418/Tensordot/concat/axis?
(discriminator/dense_418/Tensordot/concatConcatV2/discriminator/dense_418/Tensordot/free:output:0/discriminator/dense_418/Tensordot/axes:output:06discriminator/dense_418/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_418/Tensordot/concat?
'discriminator/dense_418/Tensordot/stackPack/discriminator/dense_418/Tensordot/Prod:output:01discriminator/dense_418/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_418/Tensordot/stack?
+discriminator/dense_418/Tensordot/transpose	Transpose discriminator/dense_417/Tanh:y:01discriminator/dense_418/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_418/Tensordot/transpose?
)discriminator/dense_418/Tensordot/ReshapeReshape/discriminator/dense_418/Tensordot/transpose:y:00discriminator/dense_418/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_418/Tensordot/Reshape?
(discriminator/dense_418/Tensordot/MatMulMatMul2discriminator/dense_418/Tensordot/Reshape:output:08discriminator/dense_418/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_418/Tensordot/MatMul?
)discriminator/dense_418/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_418/Tensordot/Const_2?
/discriminator/dense_418/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_418/Tensordot/concat_1/axis?
*discriminator/dense_418/Tensordot/concat_1ConcatV23discriminator/dense_418/Tensordot/GatherV2:output:02discriminator/dense_418/Tensordot/Const_2:output:08discriminator/dense_418/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_418/Tensordot/concat_1?
!discriminator/dense_418/TensordotReshape2discriminator/dense_418/Tensordot/MatMul:product:03discriminator/dense_418/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_418/Tensordot?
.discriminator/dense_418/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_418_biasadd_readvariableop_dense_418_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_418/BiasAdd/ReadVariableOp?
discriminator/dense_418/BiasAddBiasAdd*discriminator/dense_418/Tensordot:output:06discriminator/dense_418/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_418/BiasAdd?
discriminator/dense_418/TanhTanh(discriminator/dense_418/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_418/Tanh?
0discriminator/dense_419/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_419_tensordot_readvariableop_dense_419_kernel*
_output_shapes
:	?*
dtype022
0discriminator/dense_419/Tensordot/ReadVariableOp?
&discriminator/dense_419/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_419/Tensordot/axes?
&discriminator/dense_419/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_419/Tensordot/free?
'discriminator/dense_419/Tensordot/ShapeShape discriminator/dense_418/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_419/Tensordot/Shape?
/discriminator/dense_419/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_419/Tensordot/GatherV2/axis?
*discriminator/dense_419/Tensordot/GatherV2GatherV20discriminator/dense_419/Tensordot/Shape:output:0/discriminator/dense_419/Tensordot/free:output:08discriminator/dense_419/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_419/Tensordot/GatherV2?
1discriminator/dense_419/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_419/Tensordot/GatherV2_1/axis?
,discriminator/dense_419/Tensordot/GatherV2_1GatherV20discriminator/dense_419/Tensordot/Shape:output:0/discriminator/dense_419/Tensordot/axes:output:0:discriminator/dense_419/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_419/Tensordot/GatherV2_1?
'discriminator/dense_419/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_419/Tensordot/Const?
&discriminator/dense_419/Tensordot/ProdProd3discriminator/dense_419/Tensordot/GatherV2:output:00discriminator/dense_419/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_419/Tensordot/Prod?
)discriminator/dense_419/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_419/Tensordot/Const_1?
(discriminator/dense_419/Tensordot/Prod_1Prod5discriminator/dense_419/Tensordot/GatherV2_1:output:02discriminator/dense_419/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_419/Tensordot/Prod_1?
-discriminator/dense_419/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_419/Tensordot/concat/axis?
(discriminator/dense_419/Tensordot/concatConcatV2/discriminator/dense_419/Tensordot/free:output:0/discriminator/dense_419/Tensordot/axes:output:06discriminator/dense_419/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_419/Tensordot/concat?
'discriminator/dense_419/Tensordot/stackPack/discriminator/dense_419/Tensordot/Prod:output:01discriminator/dense_419/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_419/Tensordot/stack?
+discriminator/dense_419/Tensordot/transpose	Transpose discriminator/dense_418/Tanh:y:01discriminator/dense_419/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_419/Tensordot/transpose?
)discriminator/dense_419/Tensordot/ReshapeReshape/discriminator/dense_419/Tensordot/transpose:y:00discriminator/dense_419/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_419/Tensordot/Reshape?
(discriminator/dense_419/Tensordot/MatMulMatMul2discriminator/dense_419/Tensordot/Reshape:output:08discriminator/dense_419/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(discriminator/dense_419/Tensordot/MatMul?
)discriminator/dense_419/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)discriminator/dense_419/Tensordot/Const_2?
/discriminator/dense_419/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_419/Tensordot/concat_1/axis?
*discriminator/dense_419/Tensordot/concat_1ConcatV23discriminator/dense_419/Tensordot/GatherV2:output:02discriminator/dense_419/Tensordot/Const_2:output:08discriminator/dense_419/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_419/Tensordot/concat_1?
!discriminator/dense_419/TensordotReshape2discriminator/dense_419/Tensordot/MatMul:product:03discriminator/dense_419/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2#
!discriminator/dense_419/Tensordot?
.discriminator/dense_419/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_419_biasadd_readvariableop_dense_419_bias*
_output_shapes
:*
dtype020
.discriminator/dense_419/BiasAdd/ReadVariableOp?
discriminator/dense_419/BiasAddBiasAdd*discriminator/dense_419/Tensordot:output:06discriminator/dense_419/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2!
discriminator/dense_419/BiasAdd?
discriminator/dense_419/TanhTanh(discriminator/dense_419/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
discriminator/dense_419/Tanh?
discriminator/flatten_51/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2 
discriminator/flatten_51/Const?
 discriminator/flatten_51/ReshapeReshape discriminator/dense_419/Tanh:y:0'discriminator/flatten_51/Const:output:0*
T0*'
_output_shapes
:?????????2"
 discriminator/flatten_51/Reshape?
-discriminator/dense_420/MatMul/ReadVariableOpReadVariableOp>discriminator_dense_420_matmul_readvariableop_dense_420_kernel*
_output_shapes

:*
dtype02/
-discriminator/dense_420/MatMul/ReadVariableOp?
discriminator/dense_420/MatMulMatMul)discriminator/flatten_51/Reshape:output:05discriminator/dense_420/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
discriminator/dense_420/MatMul?
.discriminator/dense_420/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_420_biasadd_readvariableop_dense_420_bias*
_output_shapes
:*
dtype020
.discriminator/dense_420/BiasAdd/ReadVariableOp?
discriminator/dense_420/BiasAddBiasAdd(discriminator/dense_420/MatMul:product:06discriminator/dense_420/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
discriminator/dense_420/BiasAdd?
discriminator/dense_420/SigmoidSigmoid(discriminator/dense_420/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2!
discriminator/dense_420/Sigmoid?
IdentityIdentity#discriminator/dense_420/Sigmoid:y:0-^autoencoder/dense_412/BiasAdd/ReadVariableOp/^autoencoder/dense_412/Tensordot/ReadVariableOp-^autoencoder/dense_413/BiasAdd/ReadVariableOp/^autoencoder/dense_413/Tensordot/ReadVariableOp-^autoencoder/dense_414/BiasAdd/ReadVariableOp/^autoencoder/dense_414/Tensordot/ReadVariableOp/^discriminator/dense_415/BiasAdd/ReadVariableOp1^discriminator/dense_415/Tensordot/ReadVariableOp/^discriminator/dense_416/BiasAdd/ReadVariableOp1^discriminator/dense_416/Tensordot/ReadVariableOp/^discriminator/dense_417/BiasAdd/ReadVariableOp1^discriminator/dense_417/Tensordot/ReadVariableOp/^discriminator/dense_418/BiasAdd/ReadVariableOp1^discriminator/dense_418/Tensordot/ReadVariableOp/^discriminator/dense_419/BiasAdd/ReadVariableOp1^discriminator/dense_419/Tensordot/ReadVariableOp/^discriminator/dense_420/BiasAdd/ReadVariableOp.^discriminator/dense_420/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::2\
,autoencoder/dense_412/BiasAdd/ReadVariableOp,autoencoder/dense_412/BiasAdd/ReadVariableOp2`
.autoencoder/dense_412/Tensordot/ReadVariableOp.autoencoder/dense_412/Tensordot/ReadVariableOp2\
,autoencoder/dense_413/BiasAdd/ReadVariableOp,autoencoder/dense_413/BiasAdd/ReadVariableOp2`
.autoencoder/dense_413/Tensordot/ReadVariableOp.autoencoder/dense_413/Tensordot/ReadVariableOp2\
,autoencoder/dense_414/BiasAdd/ReadVariableOp,autoencoder/dense_414/BiasAdd/ReadVariableOp2`
.autoencoder/dense_414/Tensordot/ReadVariableOp.autoencoder/dense_414/Tensordot/ReadVariableOp2`
.discriminator/dense_415/BiasAdd/ReadVariableOp.discriminator/dense_415/BiasAdd/ReadVariableOp2d
0discriminator/dense_415/Tensordot/ReadVariableOp0discriminator/dense_415/Tensordot/ReadVariableOp2`
.discriminator/dense_416/BiasAdd/ReadVariableOp.discriminator/dense_416/BiasAdd/ReadVariableOp2d
0discriminator/dense_416/Tensordot/ReadVariableOp0discriminator/dense_416/Tensordot/ReadVariableOp2`
.discriminator/dense_417/BiasAdd/ReadVariableOp.discriminator/dense_417/BiasAdd/ReadVariableOp2d
0discriminator/dense_417/Tensordot/ReadVariableOp0discriminator/dense_417/Tensordot/ReadVariableOp2`
.discriminator/dense_418/BiasAdd/ReadVariableOp.discriminator/dense_418/BiasAdd/ReadVariableOp2d
0discriminator/dense_418/Tensordot/ReadVariableOp0discriminator/dense_418/Tensordot/ReadVariableOp2`
.discriminator/dense_419/BiasAdd/ReadVariableOp.discriminator/dense_419/BiasAdd/ReadVariableOp2d
0discriminator/dense_419/Tensordot/ReadVariableOp0discriminator/dense_419/Tensordot/ReadVariableOp2`
.discriminator/dense_420/BiasAdd/ReadVariableOp.discriminator/dense_420/BiasAdd/ReadVariableOp2^
-discriminator/dense_420/MatMul/ReadVariableOp-discriminator/dense_420/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_416_layer_call_and_return_conditional_losses_60990693

inputs-
)tensordot_readvariableop_dense_416_kernel)
%biasadd_readvariableop_dense_416_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_416_kernel* 
_output_shapes
:
??*
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_416_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_412_layer_call_fn_60990522

inputs
dense_412_kernel
dense_412_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_412_kerneldense_412_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_412_layer_call_and_return_conditional_losses_609883072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_model_960_layer_call_and_return_conditional_losses_60989110
input_11 
autoencoder_dense_412_kernel
autoencoder_dense_412_bias 
autoencoder_dense_413_kernel
autoencoder_dense_413_bias 
autoencoder_dense_414_kernel
autoencoder_dense_414_bias"
discriminator_dense_415_kernel 
discriminator_dense_415_bias"
discriminator_dense_416_kernel 
discriminator_dense_416_bias"
discriminator_dense_417_kernel 
discriminator_dense_417_bias"
discriminator_dense_418_kernel 
discriminator_dense_418_bias"
discriminator_dense_419_kernel 
discriminator_dense_419_bias"
discriminator_dense_420_kernel 
discriminator_dense_420_bias
identity??#autoencoder/StatefulPartitionedCall?%discriminator/StatefulPartitionedCall?
#autoencoder/StatefulPartitionedCallStatefulPartitionedCallinput_11autoencoder_dense_412_kernelautoencoder_dense_412_biasautoencoder_dense_413_kernelautoencoder_dense_413_biasautoencoder_dense_414_kernelautoencoder_dense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609889482%
#autoencoder/StatefulPartitionedCall?
%discriminator/StatefulPartitionedCallStatefulPartitionedCall,autoencoder/StatefulPartitionedCall:output:0discriminator_dense_415_kerneldiscriminator_dense_415_biasdiscriminator_dense_416_kerneldiscriminator_dense_416_biasdiscriminator_dense_417_kerneldiscriminator_dense_417_biasdiscriminator_dense_418_kerneldiscriminator_dense_418_biasdiscriminator_dense_419_kerneldiscriminator_dense_419_biasdiscriminator_dense_420_kerneldiscriminator_dense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609888052'
%discriminator/StatefulPartitionedCall?
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0$^autoencoder/StatefulPartitionedCall&^discriminator/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::2J
#autoencoder/StatefulPartitionedCall#autoencoder/StatefulPartitionedCall2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
? 
?
G__inference_dense_413_layer_call_and_return_conditional_losses_60988350

inputs-
)tensordot_readvariableop_dense_413_kernel)
%biasadd_readvariableop_dense_413_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_413_kernel*
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
:????????? 2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_dense_414_layer_call_and_return_conditional_losses_60988392

inputs-
)tensordot_readvariableop_dense_414_kernel)
%biasadd_readvariableop_dense_414_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_414_kernel* 
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_416_layer_call_and_return_conditional_losses_60988575

inputs-
)tensordot_readvariableop_dense_416_kernel)
%biasadd_readvariableop_dense_416_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_416_kernel* 
_output_shapes
:
??*
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_416_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
K__inference_discriminator_layer_call_and_return_conditional_losses_60988754
input_12
dense_415_dense_415_kernel
dense_415_dense_415_bias
dense_416_dense_416_kernel
dense_416_dense_416_bias
dense_417_dense_417_kernel
dense_417_dense_417_bias
dense_418_dense_418_kernel
dense_418_dense_418_bias
dense_419_dense_419_kernel
dense_419_dense_419_bias
dense_420_dense_420_kernel
dense_420_dense_420_bias
identity??!dense_415/StatefulPartitionedCall?!dense_416/StatefulPartitionedCall?!dense_417/StatefulPartitionedCall?!dense_418/StatefulPartitionedCall?!dense_419/StatefulPartitionedCall?!dense_420/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinput_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_609884832!
dropout/StatefulPartitionedCall?
!dense_415/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_415_dense_415_kerneldense_415_dense_415_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_415_layer_call_and_return_conditional_losses_609885322#
!dense_415/StatefulPartitionedCall?
!dense_416/StatefulPartitionedCallStatefulPartitionedCall*dense_415/StatefulPartitionedCall:output:0dense_416_dense_416_kerneldense_416_dense_416_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_416_layer_call_and_return_conditional_losses_609885752#
!dense_416/StatefulPartitionedCall?
!dense_417/StatefulPartitionedCallStatefulPartitionedCall*dense_416/StatefulPartitionedCall:output:0dense_417_dense_417_kerneldense_417_dense_417_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_417_layer_call_and_return_conditional_losses_609886182#
!dense_417/StatefulPartitionedCall?
!dense_418/StatefulPartitionedCallStatefulPartitionedCall*dense_417/StatefulPartitionedCall:output:0dense_418_dense_418_kerneldense_418_dense_418_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_418_layer_call_and_return_conditional_losses_609886612#
!dense_418/StatefulPartitionedCall?
!dense_419/StatefulPartitionedCallStatefulPartitionedCall*dense_418/StatefulPartitionedCall:output:0dense_419_dense_419_kerneldense_419_dense_419_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_419_layer_call_and_return_conditional_losses_609887042#
!dense_419/StatefulPartitionedCall?
flatten_51/PartitionedCallPartitionedCall*dense_419/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_51_layer_call_and_return_conditional_losses_609887222
flatten_51/PartitionedCall?
!dense_420/StatefulPartitionedCallStatefulPartitionedCall#flatten_51/PartitionedCall:output:0dense_420_dense_420_kerneldense_420_dense_420_bias*
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
G__inference_dense_420_layer_call_and_return_conditional_losses_609887412#
!dense_420/StatefulPartitionedCall?
IdentityIdentity*dense_420/StatefulPartitionedCall:output:0"^dense_415/StatefulPartitionedCall"^dense_416/StatefulPartitionedCall"^dense_417/StatefulPartitionedCall"^dense_418/StatefulPartitionedCall"^dense_419/StatefulPartitionedCall"^dense_420/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::2F
!dense_415/StatefulPartitionedCall!dense_415/StatefulPartitionedCall2F
!dense_416/StatefulPartitionedCall!dense_416/StatefulPartitionedCall2F
!dense_417/StatefulPartitionedCall!dense_417/StatefulPartitionedCall2F
!dense_418/StatefulPartitionedCall!dense_418/StatefulPartitionedCall2F
!dense_419/StatefulPartitionedCall!dense_419/StatefulPartitionedCall2F
!dense_420/StatefulPartitionedCall!dense_420/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_12
? 
?
G__inference_dense_418_layer_call_and_return_conditional_losses_60988661

inputs-
)tensordot_readvariableop_dense_418_kernel)
%biasadd_readvariableop_dense_418_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_418_kernel* 
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_418_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_model_960_layer_call_fn_60989182
input_11
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11dense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_biasdense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_960_layer_call_and_return_conditional_losses_609891612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
?
d
H__inference_flatten_51_layer_call_and_return_conditional_losses_60990820

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_419_layer_call_and_return_conditional_losses_60990807

inputs-
)tensordot_readvariableop_dense_419_kernel)
%biasadd_readvariableop_dense_419_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_419_kernel*
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
:??????????2
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
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_419_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_413_layer_call_and_return_conditional_losses_60990553

inputs-
)tensordot_readvariableop_dense_413_kernel)
%biasadd_readvariableop_dense_413_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_413_kernel*
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
:????????? 2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_model_960_layer_call_and_return_conditional_losses_60989208

inputs 
autoencoder_dense_412_kernel
autoencoder_dense_412_bias 
autoencoder_dense_413_kernel
autoencoder_dense_413_bias 
autoencoder_dense_414_kernel
autoencoder_dense_414_bias"
discriminator_dense_415_kernel 
discriminator_dense_415_bias"
discriminator_dense_416_kernel 
discriminator_dense_416_bias"
discriminator_dense_417_kernel 
discriminator_dense_417_bias"
discriminator_dense_418_kernel 
discriminator_dense_418_bias"
discriminator_dense_419_kernel 
discriminator_dense_419_bias"
discriminator_dense_420_kernel 
discriminator_dense_420_bias
identity??#autoencoder/StatefulPartitionedCall?%discriminator/StatefulPartitionedCall?
#autoencoder/StatefulPartitionedCallStatefulPartitionedCallinputsautoencoder_dense_412_kernelautoencoder_dense_412_biasautoencoder_dense_413_kernelautoencoder_dense_413_biasautoencoder_dense_414_kernelautoencoder_dense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609890322%
#autoencoder/StatefulPartitionedCall?
%discriminator/StatefulPartitionedCallStatefulPartitionedCall,autoencoder/StatefulPartitionedCall:output:0discriminator_dense_415_kerneldiscriminator_dense_415_biasdiscriminator_dense_416_kerneldiscriminator_dense_416_biasdiscriminator_dense_417_kerneldiscriminator_dense_417_biasdiscriminator_dense_418_kerneldiscriminator_dense_418_biasdiscriminator_dense_419_kerneldiscriminator_dense_419_biasdiscriminator_dense_420_kerneldiscriminator_dense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609888462'
%discriminator/StatefulPartitionedCall?
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0$^autoencoder/StatefulPartitionedCall&^discriminator/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::2J
#autoencoder/StatefulPartitionedCall#autoencoder/StatefulPartitionedCall2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_60989944

inputs
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609884342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_60988467
input_11
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11dense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609884582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
??
?	
K__inference_discriminator_layer_call_and_return_conditional_losses_60990450

inputs7
3dense_415_tensordot_readvariableop_dense_415_kernel3
/dense_415_biasadd_readvariableop_dense_415_bias7
3dense_416_tensordot_readvariableop_dense_416_kernel3
/dense_416_biasadd_readvariableop_dense_416_bias7
3dense_417_tensordot_readvariableop_dense_417_kernel3
/dense_417_biasadd_readvariableop_dense_417_bias7
3dense_418_tensordot_readvariableop_dense_418_kernel3
/dense_418_biasadd_readvariableop_dense_418_bias7
3dense_419_tensordot_readvariableop_dense_419_kernel3
/dense_419_biasadd_readvariableop_dense_419_bias4
0dense_420_matmul_readvariableop_dense_420_kernel3
/dense_420_biasadd_readvariableop_dense_420_bias
identity?? dense_415/BiasAdd/ReadVariableOp?"dense_415/Tensordot/ReadVariableOp? dense_416/BiasAdd/ReadVariableOp?"dense_416/Tensordot/ReadVariableOp? dense_417/BiasAdd/ReadVariableOp?"dense_417/Tensordot/ReadVariableOp? dense_418/BiasAdd/ReadVariableOp?"dense_418/Tensordot/ReadVariableOp? dense_419/BiasAdd/ReadVariableOp?"dense_419/Tensordot/ReadVariableOp? dense_420/BiasAdd/ReadVariableOp?dense_420/MatMul/ReadVariableOpo
dropout/IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2
dropout/Identity?
"dense_415/Tensordot/ReadVariableOpReadVariableOp3dense_415_tensordot_readvariableop_dense_415_kernel* 
_output_shapes
:
??*
dtype02$
"dense_415/Tensordot/ReadVariableOp~
dense_415/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_415/Tensordot/axes?
dense_415/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_415/Tensordot/free
dense_415/Tensordot/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:2
dense_415/Tensordot/Shape?
!dense_415/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_415/Tensordot/GatherV2/axis?
dense_415/Tensordot/GatherV2GatherV2"dense_415/Tensordot/Shape:output:0!dense_415/Tensordot/free:output:0*dense_415/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_415/Tensordot/GatherV2?
#dense_415/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_415/Tensordot/GatherV2_1/axis?
dense_415/Tensordot/GatherV2_1GatherV2"dense_415/Tensordot/Shape:output:0!dense_415/Tensordot/axes:output:0,dense_415/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_415/Tensordot/GatherV2_1?
dense_415/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_415/Tensordot/Const?
dense_415/Tensordot/ProdProd%dense_415/Tensordot/GatherV2:output:0"dense_415/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_415/Tensordot/Prod?
dense_415/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_415/Tensordot/Const_1?
dense_415/Tensordot/Prod_1Prod'dense_415/Tensordot/GatherV2_1:output:0$dense_415/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_415/Tensordot/Prod_1?
dense_415/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_415/Tensordot/concat/axis?
dense_415/Tensordot/concatConcatV2!dense_415/Tensordot/free:output:0!dense_415/Tensordot/axes:output:0(dense_415/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_415/Tensordot/concat?
dense_415/Tensordot/stackPack!dense_415/Tensordot/Prod:output:0#dense_415/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_415/Tensordot/stack?
dense_415/Tensordot/transpose	Transposedropout/Identity:output:0#dense_415/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_415/Tensordot/transpose?
dense_415/Tensordot/ReshapeReshape!dense_415/Tensordot/transpose:y:0"dense_415/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_415/Tensordot/Reshape?
dense_415/Tensordot/MatMulMatMul$dense_415/Tensordot/Reshape:output:0*dense_415/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_415/Tensordot/MatMul?
dense_415/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_415/Tensordot/Const_2?
!dense_415/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_415/Tensordot/concat_1/axis?
dense_415/Tensordot/concat_1ConcatV2%dense_415/Tensordot/GatherV2:output:0$dense_415/Tensordot/Const_2:output:0*dense_415/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_415/Tensordot/concat_1?
dense_415/TensordotReshape$dense_415/Tensordot/MatMul:product:0%dense_415/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_415/Tensordot?
 dense_415/BiasAdd/ReadVariableOpReadVariableOp/dense_415_biasadd_readvariableop_dense_415_bias*
_output_shapes	
:?*
dtype02"
 dense_415/BiasAdd/ReadVariableOp?
dense_415/BiasAddBiasAdddense_415/Tensordot:output:0(dense_415/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_415/BiasAdd{
dense_415/TanhTanhdense_415/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_415/Tanh?
"dense_416/Tensordot/ReadVariableOpReadVariableOp3dense_416_tensordot_readvariableop_dense_416_kernel* 
_output_shapes
:
??*
dtype02$
"dense_416/Tensordot/ReadVariableOp~
dense_416/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_416/Tensordot/axes?
dense_416/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_416/Tensordot/freex
dense_416/Tensordot/ShapeShapedense_415/Tanh:y:0*
T0*
_output_shapes
:2
dense_416/Tensordot/Shape?
!dense_416/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_416/Tensordot/GatherV2/axis?
dense_416/Tensordot/GatherV2GatherV2"dense_416/Tensordot/Shape:output:0!dense_416/Tensordot/free:output:0*dense_416/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_416/Tensordot/GatherV2?
#dense_416/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_416/Tensordot/GatherV2_1/axis?
dense_416/Tensordot/GatherV2_1GatherV2"dense_416/Tensordot/Shape:output:0!dense_416/Tensordot/axes:output:0,dense_416/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_416/Tensordot/GatherV2_1?
dense_416/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_416/Tensordot/Const?
dense_416/Tensordot/ProdProd%dense_416/Tensordot/GatherV2:output:0"dense_416/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_416/Tensordot/Prod?
dense_416/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_416/Tensordot/Const_1?
dense_416/Tensordot/Prod_1Prod'dense_416/Tensordot/GatherV2_1:output:0$dense_416/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_416/Tensordot/Prod_1?
dense_416/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_416/Tensordot/concat/axis?
dense_416/Tensordot/concatConcatV2!dense_416/Tensordot/free:output:0!dense_416/Tensordot/axes:output:0(dense_416/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_416/Tensordot/concat?
dense_416/Tensordot/stackPack!dense_416/Tensordot/Prod:output:0#dense_416/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_416/Tensordot/stack?
dense_416/Tensordot/transpose	Transposedense_415/Tanh:y:0#dense_416/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_416/Tensordot/transpose?
dense_416/Tensordot/ReshapeReshape!dense_416/Tensordot/transpose:y:0"dense_416/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_416/Tensordot/Reshape?
dense_416/Tensordot/MatMulMatMul$dense_416/Tensordot/Reshape:output:0*dense_416/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_416/Tensordot/MatMul?
dense_416/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_416/Tensordot/Const_2?
!dense_416/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_416/Tensordot/concat_1/axis?
dense_416/Tensordot/concat_1ConcatV2%dense_416/Tensordot/GatherV2:output:0$dense_416/Tensordot/Const_2:output:0*dense_416/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_416/Tensordot/concat_1?
dense_416/TensordotReshape$dense_416/Tensordot/MatMul:product:0%dense_416/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_416/Tensordot?
 dense_416/BiasAdd/ReadVariableOpReadVariableOp/dense_416_biasadd_readvariableop_dense_416_bias*
_output_shapes	
:?*
dtype02"
 dense_416/BiasAdd/ReadVariableOp?
dense_416/BiasAddBiasAdddense_416/Tensordot:output:0(dense_416/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_416/BiasAdd{
dense_416/TanhTanhdense_416/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_416/Tanh?
"dense_417/Tensordot/ReadVariableOpReadVariableOp3dense_417_tensordot_readvariableop_dense_417_kernel* 
_output_shapes
:
??*
dtype02$
"dense_417/Tensordot/ReadVariableOp~
dense_417/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_417/Tensordot/axes?
dense_417/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_417/Tensordot/freex
dense_417/Tensordot/ShapeShapedense_416/Tanh:y:0*
T0*
_output_shapes
:2
dense_417/Tensordot/Shape?
!dense_417/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_417/Tensordot/GatherV2/axis?
dense_417/Tensordot/GatherV2GatherV2"dense_417/Tensordot/Shape:output:0!dense_417/Tensordot/free:output:0*dense_417/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_417/Tensordot/GatherV2?
#dense_417/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_417/Tensordot/GatherV2_1/axis?
dense_417/Tensordot/GatherV2_1GatherV2"dense_417/Tensordot/Shape:output:0!dense_417/Tensordot/axes:output:0,dense_417/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_417/Tensordot/GatherV2_1?
dense_417/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_417/Tensordot/Const?
dense_417/Tensordot/ProdProd%dense_417/Tensordot/GatherV2:output:0"dense_417/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_417/Tensordot/Prod?
dense_417/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_417/Tensordot/Const_1?
dense_417/Tensordot/Prod_1Prod'dense_417/Tensordot/GatherV2_1:output:0$dense_417/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_417/Tensordot/Prod_1?
dense_417/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_417/Tensordot/concat/axis?
dense_417/Tensordot/concatConcatV2!dense_417/Tensordot/free:output:0!dense_417/Tensordot/axes:output:0(dense_417/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_417/Tensordot/concat?
dense_417/Tensordot/stackPack!dense_417/Tensordot/Prod:output:0#dense_417/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_417/Tensordot/stack?
dense_417/Tensordot/transpose	Transposedense_416/Tanh:y:0#dense_417/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_417/Tensordot/transpose?
dense_417/Tensordot/ReshapeReshape!dense_417/Tensordot/transpose:y:0"dense_417/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_417/Tensordot/Reshape?
dense_417/Tensordot/MatMulMatMul$dense_417/Tensordot/Reshape:output:0*dense_417/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_417/Tensordot/MatMul?
dense_417/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_417/Tensordot/Const_2?
!dense_417/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_417/Tensordot/concat_1/axis?
dense_417/Tensordot/concat_1ConcatV2%dense_417/Tensordot/GatherV2:output:0$dense_417/Tensordot/Const_2:output:0*dense_417/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_417/Tensordot/concat_1?
dense_417/TensordotReshape$dense_417/Tensordot/MatMul:product:0%dense_417/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_417/Tensordot?
 dense_417/BiasAdd/ReadVariableOpReadVariableOp/dense_417_biasadd_readvariableop_dense_417_bias*
_output_shapes	
:?*
dtype02"
 dense_417/BiasAdd/ReadVariableOp?
dense_417/BiasAddBiasAdddense_417/Tensordot:output:0(dense_417/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_417/BiasAdd{
dense_417/TanhTanhdense_417/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_417/Tanh?
"dense_418/Tensordot/ReadVariableOpReadVariableOp3dense_418_tensordot_readvariableop_dense_418_kernel* 
_output_shapes
:
??*
dtype02$
"dense_418/Tensordot/ReadVariableOp~
dense_418/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_418/Tensordot/axes?
dense_418/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_418/Tensordot/freex
dense_418/Tensordot/ShapeShapedense_417/Tanh:y:0*
T0*
_output_shapes
:2
dense_418/Tensordot/Shape?
!dense_418/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_418/Tensordot/GatherV2/axis?
dense_418/Tensordot/GatherV2GatherV2"dense_418/Tensordot/Shape:output:0!dense_418/Tensordot/free:output:0*dense_418/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_418/Tensordot/GatherV2?
#dense_418/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_418/Tensordot/GatherV2_1/axis?
dense_418/Tensordot/GatherV2_1GatherV2"dense_418/Tensordot/Shape:output:0!dense_418/Tensordot/axes:output:0,dense_418/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_418/Tensordot/GatherV2_1?
dense_418/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_418/Tensordot/Const?
dense_418/Tensordot/ProdProd%dense_418/Tensordot/GatherV2:output:0"dense_418/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_418/Tensordot/Prod?
dense_418/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_418/Tensordot/Const_1?
dense_418/Tensordot/Prod_1Prod'dense_418/Tensordot/GatherV2_1:output:0$dense_418/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_418/Tensordot/Prod_1?
dense_418/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_418/Tensordot/concat/axis?
dense_418/Tensordot/concatConcatV2!dense_418/Tensordot/free:output:0!dense_418/Tensordot/axes:output:0(dense_418/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_418/Tensordot/concat?
dense_418/Tensordot/stackPack!dense_418/Tensordot/Prod:output:0#dense_418/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_418/Tensordot/stack?
dense_418/Tensordot/transpose	Transposedense_417/Tanh:y:0#dense_418/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_418/Tensordot/transpose?
dense_418/Tensordot/ReshapeReshape!dense_418/Tensordot/transpose:y:0"dense_418/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_418/Tensordot/Reshape?
dense_418/Tensordot/MatMulMatMul$dense_418/Tensordot/Reshape:output:0*dense_418/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_418/Tensordot/MatMul?
dense_418/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_418/Tensordot/Const_2?
!dense_418/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_418/Tensordot/concat_1/axis?
dense_418/Tensordot/concat_1ConcatV2%dense_418/Tensordot/GatherV2:output:0$dense_418/Tensordot/Const_2:output:0*dense_418/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_418/Tensordot/concat_1?
dense_418/TensordotReshape$dense_418/Tensordot/MatMul:product:0%dense_418/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_418/Tensordot?
 dense_418/BiasAdd/ReadVariableOpReadVariableOp/dense_418_biasadd_readvariableop_dense_418_bias*
_output_shapes	
:?*
dtype02"
 dense_418/BiasAdd/ReadVariableOp?
dense_418/BiasAddBiasAdddense_418/Tensordot:output:0(dense_418/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_418/BiasAdd{
dense_418/TanhTanhdense_418/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_418/Tanh?
"dense_419/Tensordot/ReadVariableOpReadVariableOp3dense_419_tensordot_readvariableop_dense_419_kernel*
_output_shapes
:	?*
dtype02$
"dense_419/Tensordot/ReadVariableOp~
dense_419/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_419/Tensordot/axes?
dense_419/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_419/Tensordot/freex
dense_419/Tensordot/ShapeShapedense_418/Tanh:y:0*
T0*
_output_shapes
:2
dense_419/Tensordot/Shape?
!dense_419/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_419/Tensordot/GatherV2/axis?
dense_419/Tensordot/GatherV2GatherV2"dense_419/Tensordot/Shape:output:0!dense_419/Tensordot/free:output:0*dense_419/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_419/Tensordot/GatherV2?
#dense_419/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_419/Tensordot/GatherV2_1/axis?
dense_419/Tensordot/GatherV2_1GatherV2"dense_419/Tensordot/Shape:output:0!dense_419/Tensordot/axes:output:0,dense_419/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_419/Tensordot/GatherV2_1?
dense_419/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_419/Tensordot/Const?
dense_419/Tensordot/ProdProd%dense_419/Tensordot/GatherV2:output:0"dense_419/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_419/Tensordot/Prod?
dense_419/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_419/Tensordot/Const_1?
dense_419/Tensordot/Prod_1Prod'dense_419/Tensordot/GatherV2_1:output:0$dense_419/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_419/Tensordot/Prod_1?
dense_419/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_419/Tensordot/concat/axis?
dense_419/Tensordot/concatConcatV2!dense_419/Tensordot/free:output:0!dense_419/Tensordot/axes:output:0(dense_419/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_419/Tensordot/concat?
dense_419/Tensordot/stackPack!dense_419/Tensordot/Prod:output:0#dense_419/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_419/Tensordot/stack?
dense_419/Tensordot/transpose	Transposedense_418/Tanh:y:0#dense_419/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_419/Tensordot/transpose?
dense_419/Tensordot/ReshapeReshape!dense_419/Tensordot/transpose:y:0"dense_419/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_419/Tensordot/Reshape?
dense_419/Tensordot/MatMulMatMul$dense_419/Tensordot/Reshape:output:0*dense_419/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_419/Tensordot/MatMul?
dense_419/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_419/Tensordot/Const_2?
!dense_419/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_419/Tensordot/concat_1/axis?
dense_419/Tensordot/concat_1ConcatV2%dense_419/Tensordot/GatherV2:output:0$dense_419/Tensordot/Const_2:output:0*dense_419/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_419/Tensordot/concat_1?
dense_419/TensordotReshape$dense_419/Tensordot/MatMul:product:0%dense_419/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_419/Tensordot?
 dense_419/BiasAdd/ReadVariableOpReadVariableOp/dense_419_biasadd_readvariableop_dense_419_bias*
_output_shapes
:*
dtype02"
 dense_419/BiasAdd/ReadVariableOp?
dense_419/BiasAddBiasAdddense_419/Tensordot:output:0(dense_419/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_419/BiasAddz
dense_419/TanhTanhdense_419/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense_419/Tanhu
flatten_51/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_51/Const?
flatten_51/ReshapeReshapedense_419/Tanh:y:0flatten_51/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_51/Reshape?
dense_420/MatMul/ReadVariableOpReadVariableOp0dense_420_matmul_readvariableop_dense_420_kernel*
_output_shapes

:*
dtype02!
dense_420/MatMul/ReadVariableOp?
dense_420/MatMulMatMulflatten_51/Reshape:output:0'dense_420/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_420/MatMul?
 dense_420/BiasAdd/ReadVariableOpReadVariableOp/dense_420_biasadd_readvariableop_dense_420_bias*
_output_shapes
:*
dtype02"
 dense_420/BiasAdd/ReadVariableOp?
dense_420/BiasAddBiasAdddense_420/MatMul:product:0(dense_420/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_420/BiasAdd
dense_420/SigmoidSigmoiddense_420/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_420/Sigmoid?
IdentityIdentitydense_420/Sigmoid:y:0!^dense_415/BiasAdd/ReadVariableOp#^dense_415/Tensordot/ReadVariableOp!^dense_416/BiasAdd/ReadVariableOp#^dense_416/Tensordot/ReadVariableOp!^dense_417/BiasAdd/ReadVariableOp#^dense_417/Tensordot/ReadVariableOp!^dense_418/BiasAdd/ReadVariableOp#^dense_418/Tensordot/ReadVariableOp!^dense_419/BiasAdd/ReadVariableOp#^dense_419/Tensordot/ReadVariableOp!^dense_420/BiasAdd/ReadVariableOp ^dense_420/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::2D
 dense_415/BiasAdd/ReadVariableOp dense_415/BiasAdd/ReadVariableOp2H
"dense_415/Tensordot/ReadVariableOp"dense_415/Tensordot/ReadVariableOp2D
 dense_416/BiasAdd/ReadVariableOp dense_416/BiasAdd/ReadVariableOp2H
"dense_416/Tensordot/ReadVariableOp"dense_416/Tensordot/ReadVariableOp2D
 dense_417/BiasAdd/ReadVariableOp dense_417/BiasAdd/ReadVariableOp2H
"dense_417/Tensordot/ReadVariableOp"dense_417/Tensordot/ReadVariableOp2D
 dense_418/BiasAdd/ReadVariableOp dense_418/BiasAdd/ReadVariableOp2H
"dense_418/Tensordot/ReadVariableOp"dense_418/Tensordot/ReadVariableOp2D
 dense_419/BiasAdd/ReadVariableOp dense_419/BiasAdd/ReadVariableOp2H
"dense_419/Tensordot/ReadVariableOp"dense_419/Tensordot/ReadVariableOp2D
 dense_420/BiasAdd/ReadVariableOp dense_420/BiasAdd/ReadVariableOp2B
dense_420/MatMul/ReadVariableOpdense_420/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_51_layer_call_and_return_conditional_losses_60988722

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_414_layer_call_fn_60990597

inputs
dense_414_kernel
dense_414_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_414_kerneldense_414_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_414_layer_call_and_return_conditional_losses_609883922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_415_layer_call_and_return_conditional_losses_60990655

inputs-
)tensordot_readvariableop_dense_415_kernel)
%biasadd_readvariableop_dense_415_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_415_kernel* 
_output_shapes
:
??*
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
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_415_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
0__inference_discriminator_layer_call_fn_60988861
input_12
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12dense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609888462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_12
?
c
E__inference_dropout_layer_call_and_return_conditional_losses_60990614

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?q
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60990123
inputs_07
3dense_412_tensordot_readvariableop_dense_412_kernel3
/dense_412_biasadd_readvariableop_dense_412_bias7
3dense_413_tensordot_readvariableop_dense_413_kernel3
/dense_413_biasadd_readvariableop_dense_413_bias7
3dense_414_tensordot_readvariableop_dense_414_kernel3
/dense_414_biasadd_readvariableop_dense_414_bias
identity?? dense_412/BiasAdd/ReadVariableOp?"dense_412/Tensordot/ReadVariableOp? dense_413/BiasAdd/ReadVariableOp?"dense_413/Tensordot/ReadVariableOp? dense_414/BiasAdd/ReadVariableOp?"dense_414/Tensordot/ReadVariableOp?
"dense_412/Tensordot/ReadVariableOpReadVariableOp3dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype02$
"dense_412/Tensordot/ReadVariableOp~
dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_412/Tensordot/axes?
dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_412/Tensordot/freen
dense_412/Tensordot/ShapeShapeinputs_0*
T0*
_output_shapes
:2
dense_412/Tensordot/Shape?
!dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/GatherV2/axis?
dense_412/Tensordot/GatherV2GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/free:output:0*dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_412/Tensordot/GatherV2?
#dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_412/Tensordot/GatherV2_1/axis?
dense_412/Tensordot/GatherV2_1GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/axes:output:0,dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_412/Tensordot/GatherV2_1?
dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const?
dense_412/Tensordot/ProdProd%dense_412/Tensordot/GatherV2:output:0"dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod?
dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_1?
dense_412/Tensordot/Prod_1Prod'dense_412/Tensordot/GatherV2_1:output:0$dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod_1?
dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_412/Tensordot/concat/axis?
dense_412/Tensordot/concatConcatV2!dense_412/Tensordot/free:output:0!dense_412/Tensordot/axes:output:0(dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat?
dense_412/Tensordot/stackPack!dense_412/Tensordot/Prod:output:0#dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/stack?
dense_412/Tensordot/transpose	Transposeinputs_0#dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_412/Tensordot/transpose?
dense_412/Tensordot/ReshapeReshape!dense_412/Tensordot/transpose:y:0"dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_412/Tensordot/Reshape?
dense_412/Tensordot/MatMulMatMul$dense_412/Tensordot/Reshape:output:0*dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_412/Tensordot/MatMul?
dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_2?
!dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/concat_1/axis?
dense_412/Tensordot/concat_1ConcatV2%dense_412/Tensordot/GatherV2:output:0$dense_412/Tensordot/Const_2:output:0*dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat_1?
dense_412/TensordotReshape$dense_412/Tensordot/MatMul:product:0%dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tensordot?
 dense_412/BiasAdd/ReadVariableOpReadVariableOp/dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02"
 dense_412/BiasAdd/ReadVariableOp?
dense_412/BiasAddBiasAdddense_412/Tensordot:output:0(dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_412/BiasAddz
dense_412/TanhTanhdense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tanh?
"dense_413/Tensordot/ReadVariableOpReadVariableOp3dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_413/Tensordot/ReadVariableOp~
dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_413/Tensordot/axes?
dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_413/Tensordot/freex
dense_413/Tensordot/ShapeShapedense_412/Tanh:y:0*
T0*
_output_shapes
:2
dense_413/Tensordot/Shape?
!dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/GatherV2/axis?
dense_413/Tensordot/GatherV2GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/free:output:0*dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_413/Tensordot/GatherV2?
#dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_413/Tensordot/GatherV2_1/axis?
dense_413/Tensordot/GatherV2_1GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/axes:output:0,dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_413/Tensordot/GatherV2_1?
dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const?
dense_413/Tensordot/ProdProd%dense_413/Tensordot/GatherV2:output:0"dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod?
dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const_1?
dense_413/Tensordot/Prod_1Prod'dense_413/Tensordot/GatherV2_1:output:0$dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod_1?
dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_413/Tensordot/concat/axis?
dense_413/Tensordot/concatConcatV2!dense_413/Tensordot/free:output:0!dense_413/Tensordot/axes:output:0(dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat?
dense_413/Tensordot/stackPack!dense_413/Tensordot/Prod:output:0#dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/stack?
dense_413/Tensordot/transpose	Transposedense_412/Tanh:y:0#dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_413/Tensordot/transpose?
dense_413/Tensordot/ReshapeReshape!dense_413/Tensordot/transpose:y:0"dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_413/Tensordot/Reshape?
dense_413/Tensordot/MatMulMatMul$dense_413/Tensordot/Reshape:output:0*dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_413/Tensordot/MatMul?
dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_413/Tensordot/Const_2?
!dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/concat_1/axis?
dense_413/Tensordot/concat_1ConcatV2%dense_413/Tensordot/GatherV2:output:0$dense_413/Tensordot/Const_2:output:0*dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat_1?
dense_413/TensordotReshape$dense_413/Tensordot/MatMul:product:0%dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tensordot?
 dense_413/BiasAdd/ReadVariableOpReadVariableOp/dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02"
 dense_413/BiasAdd/ReadVariableOp?
dense_413/BiasAddBiasAdddense_413/Tensordot:output:0(dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_413/BiasAdd{
dense_413/TanhTanhdense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tanh?
"dense_414/Tensordot/ReadVariableOpReadVariableOp3dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype02$
"dense_414/Tensordot/ReadVariableOp~
dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_414/Tensordot/axes?
dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_414/Tensordot/freex
dense_414/Tensordot/ShapeShapedense_413/Tanh:y:0*
T0*
_output_shapes
:2
dense_414/Tensordot/Shape?
!dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/GatherV2/axis?
dense_414/Tensordot/GatherV2GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/free:output:0*dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_414/Tensordot/GatherV2?
#dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_414/Tensordot/GatherV2_1/axis?
dense_414/Tensordot/GatherV2_1GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/axes:output:0,dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_414/Tensordot/GatherV2_1?
dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const?
dense_414/Tensordot/ProdProd%dense_414/Tensordot/GatherV2:output:0"dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod?
dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const_1?
dense_414/Tensordot/Prod_1Prod'dense_414/Tensordot/GatherV2_1:output:0$dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod_1?
dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_414/Tensordot/concat/axis?
dense_414/Tensordot/concatConcatV2!dense_414/Tensordot/free:output:0!dense_414/Tensordot/axes:output:0(dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat?
dense_414/Tensordot/stackPack!dense_414/Tensordot/Prod:output:0#dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/stack?
dense_414/Tensordot/transpose	Transposedense_413/Tanh:y:0#dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot/transpose?
dense_414/Tensordot/ReshapeReshape!dense_414/Tensordot/transpose:y:0"dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_414/Tensordot/Reshape?
dense_414/Tensordot/MatMulMatMul$dense_414/Tensordot/Reshape:output:0*dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_414/Tensordot/MatMul?
dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_414/Tensordot/Const_2?
!dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/concat_1/axis?
dense_414/Tensordot/concat_1ConcatV2%dense_414/Tensordot/GatherV2:output:0$dense_414/Tensordot/Const_2:output:0*dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat_1?
dense_414/TensordotReshape$dense_414/Tensordot/MatMul:product:0%dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot?
 dense_414/BiasAdd/ReadVariableOpReadVariableOp/dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02"
 dense_414/BiasAdd/ReadVariableOp?
dense_414/BiasAddBiasAdddense_414/Tensordot:output:0(dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_414/BiasAdd?
IdentityIdentitydense_414/BiasAdd:output:0!^dense_412/BiasAdd/ReadVariableOp#^dense_412/Tensordot/ReadVariableOp!^dense_413/BiasAdd/ReadVariableOp#^dense_413/Tensordot/ReadVariableOp!^dense_414/BiasAdd/ReadVariableOp#^dense_414/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_412/BiasAdd/ReadVariableOp dense_412/BiasAdd/ReadVariableOp2H
"dense_412/Tensordot/ReadVariableOp"dense_412/Tensordot/ReadVariableOp2D
 dense_413/BiasAdd/ReadVariableOp dense_413/BiasAdd/ReadVariableOp2H
"dense_413/Tensordot/ReadVariableOp"dense_413/Tensordot/ReadVariableOp2D
 dense_414/BiasAdd/ReadVariableOp dense_414/BiasAdd/ReadVariableOp2H
"dense_414/Tensordot/ReadVariableOp"dense_414/Tensordot/ReadVariableOp:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?

?
0__inference_discriminator_layer_call_fn_60990467

inputs
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609888052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_413_layer_call_fn_60990560

inputs
dense_413_kernel
dense_413_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_413_kerneldense_413_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_413_layer_call_and_return_conditional_losses_609883502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
I
-__inference_flatten_51_layer_call_fn_60990825

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_51_layer_call_and_return_conditional_losses_609887222
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_model_960_layer_call_fn_60989742

inputs
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_biasdense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_960_layer_call_and_return_conditional_losses_609891612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_419_layer_call_fn_60990814

inputs
dense_419_kernel
dense_419_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_419_kerneldense_419_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_419_layer_call_and_return_conditional_losses_609887042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
K__inference_discriminator_layer_call_and_return_conditional_losses_60988805

inputs
dense_415_dense_415_kernel
dense_415_dense_415_bias
dense_416_dense_416_kernel
dense_416_dense_416_bias
dense_417_dense_417_kernel
dense_417_dense_417_bias
dense_418_dense_418_kernel
dense_418_dense_418_bias
dense_419_dense_419_kernel
dense_419_dense_419_bias
dense_420_dense_420_kernel
dense_420_dense_420_bias
identity??!dense_415/StatefulPartitionedCall?!dense_416/StatefulPartitionedCall?!dense_417/StatefulPartitionedCall?!dense_418/StatefulPartitionedCall?!dense_419/StatefulPartitionedCall?!dense_420/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_609884832!
dropout/StatefulPartitionedCall?
!dense_415/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_415_dense_415_kerneldense_415_dense_415_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_415_layer_call_and_return_conditional_losses_609885322#
!dense_415/StatefulPartitionedCall?
!dense_416/StatefulPartitionedCallStatefulPartitionedCall*dense_415/StatefulPartitionedCall:output:0dense_416_dense_416_kerneldense_416_dense_416_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_416_layer_call_and_return_conditional_losses_609885752#
!dense_416/StatefulPartitionedCall?
!dense_417/StatefulPartitionedCallStatefulPartitionedCall*dense_416/StatefulPartitionedCall:output:0dense_417_dense_417_kerneldense_417_dense_417_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_417_layer_call_and_return_conditional_losses_609886182#
!dense_417/StatefulPartitionedCall?
!dense_418/StatefulPartitionedCallStatefulPartitionedCall*dense_417/StatefulPartitionedCall:output:0dense_418_dense_418_kerneldense_418_dense_418_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_418_layer_call_and_return_conditional_losses_609886612#
!dense_418/StatefulPartitionedCall?
!dense_419/StatefulPartitionedCallStatefulPartitionedCall*dense_418/StatefulPartitionedCall:output:0dense_419_dense_419_kerneldense_419_dense_419_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_419_layer_call_and_return_conditional_losses_609887042#
!dense_419/StatefulPartitionedCall?
flatten_51/PartitionedCallPartitionedCall*dense_419/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_51_layer_call_and_return_conditional_losses_609887222
flatten_51/PartitionedCall?
!dense_420/StatefulPartitionedCallStatefulPartitionedCall#flatten_51/PartitionedCall:output:0dense_420_dense_420_kerneldense_420_dense_420_bias*
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
G__inference_dense_420_layer_call_and_return_conditional_losses_609887412#
!dense_420/StatefulPartitionedCall?
IdentityIdentity*dense_420/StatefulPartitionedCall:output:0"^dense_415/StatefulPartitionedCall"^dense_416/StatefulPartitionedCall"^dense_417/StatefulPartitionedCall"^dense_418/StatefulPartitionedCall"^dense_419/StatefulPartitionedCall"^dense_420/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::2F
!dense_415/StatefulPartitionedCall!dense_415/StatefulPartitionedCall2F
!dense_416/StatefulPartitionedCall!dense_416/StatefulPartitionedCall2F
!dense_417/StatefulPartitionedCall!dense_417/StatefulPartitionedCall2F
!dense_418/StatefulPartitionedCall!dense_418/StatefulPartitionedCall2F
!dense_419/StatefulPartitionedCall!dense_419/StatefulPartitionedCall2F
!dense_420/StatefulPartitionedCall!dense_420/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
#__inference__wrapped_model_60988272
input_11M
Imodel_960_autoencoder_dense_412_tensordot_readvariableop_dense_412_kernelI
Emodel_960_autoencoder_dense_412_biasadd_readvariableop_dense_412_biasM
Imodel_960_autoencoder_dense_413_tensordot_readvariableop_dense_413_kernelI
Emodel_960_autoencoder_dense_413_biasadd_readvariableop_dense_413_biasM
Imodel_960_autoencoder_dense_414_tensordot_readvariableop_dense_414_kernelI
Emodel_960_autoencoder_dense_414_biasadd_readvariableop_dense_414_biasO
Kmodel_960_discriminator_dense_415_tensordot_readvariableop_dense_415_kernelK
Gmodel_960_discriminator_dense_415_biasadd_readvariableop_dense_415_biasO
Kmodel_960_discriminator_dense_416_tensordot_readvariableop_dense_416_kernelK
Gmodel_960_discriminator_dense_416_biasadd_readvariableop_dense_416_biasO
Kmodel_960_discriminator_dense_417_tensordot_readvariableop_dense_417_kernelK
Gmodel_960_discriminator_dense_417_biasadd_readvariableop_dense_417_biasO
Kmodel_960_discriminator_dense_418_tensordot_readvariableop_dense_418_kernelK
Gmodel_960_discriminator_dense_418_biasadd_readvariableop_dense_418_biasO
Kmodel_960_discriminator_dense_419_tensordot_readvariableop_dense_419_kernelK
Gmodel_960_discriminator_dense_419_biasadd_readvariableop_dense_419_biasL
Hmodel_960_discriminator_dense_420_matmul_readvariableop_dense_420_kernelK
Gmodel_960_discriminator_dense_420_biasadd_readvariableop_dense_420_bias
identity??6model_960/autoencoder/dense_412/BiasAdd/ReadVariableOp?8model_960/autoencoder/dense_412/Tensordot/ReadVariableOp?6model_960/autoencoder/dense_413/BiasAdd/ReadVariableOp?8model_960/autoencoder/dense_413/Tensordot/ReadVariableOp?6model_960/autoencoder/dense_414/BiasAdd/ReadVariableOp?8model_960/autoencoder/dense_414/Tensordot/ReadVariableOp?8model_960/discriminator/dense_415/BiasAdd/ReadVariableOp?:model_960/discriminator/dense_415/Tensordot/ReadVariableOp?8model_960/discriminator/dense_416/BiasAdd/ReadVariableOp?:model_960/discriminator/dense_416/Tensordot/ReadVariableOp?8model_960/discriminator/dense_417/BiasAdd/ReadVariableOp?:model_960/discriminator/dense_417/Tensordot/ReadVariableOp?8model_960/discriminator/dense_418/BiasAdd/ReadVariableOp?:model_960/discriminator/dense_418/Tensordot/ReadVariableOp?8model_960/discriminator/dense_419/BiasAdd/ReadVariableOp?:model_960/discriminator/dense_419/Tensordot/ReadVariableOp?8model_960/discriminator/dense_420/BiasAdd/ReadVariableOp?7model_960/discriminator/dense_420/MatMul/ReadVariableOp?
8model_960/autoencoder/dense_412/Tensordot/ReadVariableOpReadVariableOpImodel_960_autoencoder_dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype02:
8model_960/autoencoder/dense_412/Tensordot/ReadVariableOp?
.model_960/autoencoder/dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.model_960/autoencoder/dense_412/Tensordot/axes?
.model_960/autoencoder/dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.model_960/autoencoder/dense_412/Tensordot/free?
/model_960/autoencoder/dense_412/Tensordot/ShapeShapeinput_11*
T0*
_output_shapes
:21
/model_960/autoencoder/dense_412/Tensordot/Shape?
7model_960/autoencoder/dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/autoencoder/dense_412/Tensordot/GatherV2/axis?
2model_960/autoencoder/dense_412/Tensordot/GatherV2GatherV28model_960/autoencoder/dense_412/Tensordot/Shape:output:07model_960/autoencoder/dense_412/Tensordot/free:output:0@model_960/autoencoder/dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2model_960/autoencoder/dense_412/Tensordot/GatherV2?
9model_960/autoencoder/dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/autoencoder/dense_412/Tensordot/GatherV2_1/axis?
4model_960/autoencoder/dense_412/Tensordot/GatherV2_1GatherV28model_960/autoencoder/dense_412/Tensordot/Shape:output:07model_960/autoencoder/dense_412/Tensordot/axes:output:0Bmodel_960/autoencoder/dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_960/autoencoder/dense_412/Tensordot/GatherV2_1?
/model_960/autoencoder/dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_960/autoencoder/dense_412/Tensordot/Const?
.model_960/autoencoder/dense_412/Tensordot/ProdProd;model_960/autoencoder/dense_412/Tensordot/GatherV2:output:08model_960/autoencoder/dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.model_960/autoencoder/dense_412/Tensordot/Prod?
1model_960/autoencoder/dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/autoencoder/dense_412/Tensordot/Const_1?
0model_960/autoencoder/dense_412/Tensordot/Prod_1Prod=model_960/autoencoder/dense_412/Tensordot/GatherV2_1:output:0:model_960/autoencoder/dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0model_960/autoencoder/dense_412/Tensordot/Prod_1?
5model_960/autoencoder/dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5model_960/autoencoder/dense_412/Tensordot/concat/axis?
0model_960/autoencoder/dense_412/Tensordot/concatConcatV27model_960/autoencoder/dense_412/Tensordot/free:output:07model_960/autoencoder/dense_412/Tensordot/axes:output:0>model_960/autoencoder/dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0model_960/autoencoder/dense_412/Tensordot/concat?
/model_960/autoencoder/dense_412/Tensordot/stackPack7model_960/autoencoder/dense_412/Tensordot/Prod:output:09model_960/autoencoder/dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/model_960/autoencoder/dense_412/Tensordot/stack?
3model_960/autoencoder/dense_412/Tensordot/transpose	Transposeinput_119model_960/autoencoder/dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????25
3model_960/autoencoder/dense_412/Tensordot/transpose?
1model_960/autoencoder/dense_412/Tensordot/ReshapeReshape7model_960/autoencoder/dense_412/Tensordot/transpose:y:08model_960/autoencoder/dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1model_960/autoencoder/dense_412/Tensordot/Reshape?
0model_960/autoencoder/dense_412/Tensordot/MatMulMatMul:model_960/autoencoder/dense_412/Tensordot/Reshape:output:0@model_960/autoencoder/dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 22
0model_960/autoencoder/dense_412/Tensordot/MatMul?
1model_960/autoencoder/dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/autoencoder/dense_412/Tensordot/Const_2?
7model_960/autoencoder/dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/autoencoder/dense_412/Tensordot/concat_1/axis?
2model_960/autoencoder/dense_412/Tensordot/concat_1ConcatV2;model_960/autoencoder/dense_412/Tensordot/GatherV2:output:0:model_960/autoencoder/dense_412/Tensordot/Const_2:output:0@model_960/autoencoder/dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2model_960/autoencoder/dense_412/Tensordot/concat_1?
)model_960/autoencoder/dense_412/TensordotReshape:model_960/autoencoder/dense_412/Tensordot/MatMul:product:0;model_960/autoencoder/dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2+
)model_960/autoencoder/dense_412/Tensordot?
6model_960/autoencoder/dense_412/BiasAdd/ReadVariableOpReadVariableOpEmodel_960_autoencoder_dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype028
6model_960/autoencoder/dense_412/BiasAdd/ReadVariableOp?
'model_960/autoencoder/dense_412/BiasAddBiasAdd2model_960/autoencoder/dense_412/Tensordot:output:0>model_960/autoencoder/dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2)
'model_960/autoencoder/dense_412/BiasAdd?
$model_960/autoencoder/dense_412/TanhTanh0model_960/autoencoder/dense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2&
$model_960/autoencoder/dense_412/Tanh?
8model_960/autoencoder/dense_413/Tensordot/ReadVariableOpReadVariableOpImodel_960_autoencoder_dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype02:
8model_960/autoencoder/dense_413/Tensordot/ReadVariableOp?
.model_960/autoencoder/dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.model_960/autoencoder/dense_413/Tensordot/axes?
.model_960/autoencoder/dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.model_960/autoencoder/dense_413/Tensordot/free?
/model_960/autoencoder/dense_413/Tensordot/ShapeShape(model_960/autoencoder/dense_412/Tanh:y:0*
T0*
_output_shapes
:21
/model_960/autoencoder/dense_413/Tensordot/Shape?
7model_960/autoencoder/dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/autoencoder/dense_413/Tensordot/GatherV2/axis?
2model_960/autoencoder/dense_413/Tensordot/GatherV2GatherV28model_960/autoencoder/dense_413/Tensordot/Shape:output:07model_960/autoencoder/dense_413/Tensordot/free:output:0@model_960/autoencoder/dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2model_960/autoencoder/dense_413/Tensordot/GatherV2?
9model_960/autoencoder/dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/autoencoder/dense_413/Tensordot/GatherV2_1/axis?
4model_960/autoencoder/dense_413/Tensordot/GatherV2_1GatherV28model_960/autoencoder/dense_413/Tensordot/Shape:output:07model_960/autoencoder/dense_413/Tensordot/axes:output:0Bmodel_960/autoencoder/dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_960/autoencoder/dense_413/Tensordot/GatherV2_1?
/model_960/autoencoder/dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_960/autoencoder/dense_413/Tensordot/Const?
.model_960/autoencoder/dense_413/Tensordot/ProdProd;model_960/autoencoder/dense_413/Tensordot/GatherV2:output:08model_960/autoencoder/dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.model_960/autoencoder/dense_413/Tensordot/Prod?
1model_960/autoencoder/dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/autoencoder/dense_413/Tensordot/Const_1?
0model_960/autoencoder/dense_413/Tensordot/Prod_1Prod=model_960/autoencoder/dense_413/Tensordot/GatherV2_1:output:0:model_960/autoencoder/dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0model_960/autoencoder/dense_413/Tensordot/Prod_1?
5model_960/autoencoder/dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5model_960/autoencoder/dense_413/Tensordot/concat/axis?
0model_960/autoencoder/dense_413/Tensordot/concatConcatV27model_960/autoencoder/dense_413/Tensordot/free:output:07model_960/autoencoder/dense_413/Tensordot/axes:output:0>model_960/autoencoder/dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0model_960/autoencoder/dense_413/Tensordot/concat?
/model_960/autoencoder/dense_413/Tensordot/stackPack7model_960/autoencoder/dense_413/Tensordot/Prod:output:09model_960/autoencoder/dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/model_960/autoencoder/dense_413/Tensordot/stack?
3model_960/autoencoder/dense_413/Tensordot/transpose	Transpose(model_960/autoencoder/dense_412/Tanh:y:09model_960/autoencoder/dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 25
3model_960/autoencoder/dense_413/Tensordot/transpose?
1model_960/autoencoder/dense_413/Tensordot/ReshapeReshape7model_960/autoencoder/dense_413/Tensordot/transpose:y:08model_960/autoencoder/dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1model_960/autoencoder/dense_413/Tensordot/Reshape?
0model_960/autoencoder/dense_413/Tensordot/MatMulMatMul:model_960/autoencoder/dense_413/Tensordot/Reshape:output:0@model_960/autoencoder/dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0model_960/autoencoder/dense_413/Tensordot/MatMul?
1model_960/autoencoder/dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?23
1model_960/autoencoder/dense_413/Tensordot/Const_2?
7model_960/autoencoder/dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/autoencoder/dense_413/Tensordot/concat_1/axis?
2model_960/autoencoder/dense_413/Tensordot/concat_1ConcatV2;model_960/autoencoder/dense_413/Tensordot/GatherV2:output:0:model_960/autoencoder/dense_413/Tensordot/Const_2:output:0@model_960/autoencoder/dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2model_960/autoencoder/dense_413/Tensordot/concat_1?
)model_960/autoencoder/dense_413/TensordotReshape:model_960/autoencoder/dense_413/Tensordot/MatMul:product:0;model_960/autoencoder/dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2+
)model_960/autoencoder/dense_413/Tensordot?
6model_960/autoencoder/dense_413/BiasAdd/ReadVariableOpReadVariableOpEmodel_960_autoencoder_dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype028
6model_960/autoencoder/dense_413/BiasAdd/ReadVariableOp?
'model_960/autoencoder/dense_413/BiasAddBiasAdd2model_960/autoencoder/dense_413/Tensordot:output:0>model_960/autoencoder/dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2)
'model_960/autoencoder/dense_413/BiasAdd?
$model_960/autoencoder/dense_413/TanhTanh0model_960/autoencoder/dense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2&
$model_960/autoencoder/dense_413/Tanh?
8model_960/autoencoder/dense_414/Tensordot/ReadVariableOpReadVariableOpImodel_960_autoencoder_dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype02:
8model_960/autoencoder/dense_414/Tensordot/ReadVariableOp?
.model_960/autoencoder/dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.model_960/autoencoder/dense_414/Tensordot/axes?
.model_960/autoencoder/dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.model_960/autoencoder/dense_414/Tensordot/free?
/model_960/autoencoder/dense_414/Tensordot/ShapeShape(model_960/autoencoder/dense_413/Tanh:y:0*
T0*
_output_shapes
:21
/model_960/autoencoder/dense_414/Tensordot/Shape?
7model_960/autoencoder/dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/autoencoder/dense_414/Tensordot/GatherV2/axis?
2model_960/autoencoder/dense_414/Tensordot/GatherV2GatherV28model_960/autoencoder/dense_414/Tensordot/Shape:output:07model_960/autoencoder/dense_414/Tensordot/free:output:0@model_960/autoencoder/dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2model_960/autoencoder/dense_414/Tensordot/GatherV2?
9model_960/autoencoder/dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/autoencoder/dense_414/Tensordot/GatherV2_1/axis?
4model_960/autoencoder/dense_414/Tensordot/GatherV2_1GatherV28model_960/autoencoder/dense_414/Tensordot/Shape:output:07model_960/autoencoder/dense_414/Tensordot/axes:output:0Bmodel_960/autoencoder/dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_960/autoencoder/dense_414/Tensordot/GatherV2_1?
/model_960/autoencoder/dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_960/autoencoder/dense_414/Tensordot/Const?
.model_960/autoencoder/dense_414/Tensordot/ProdProd;model_960/autoencoder/dense_414/Tensordot/GatherV2:output:08model_960/autoencoder/dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.model_960/autoencoder/dense_414/Tensordot/Prod?
1model_960/autoencoder/dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/autoencoder/dense_414/Tensordot/Const_1?
0model_960/autoencoder/dense_414/Tensordot/Prod_1Prod=model_960/autoencoder/dense_414/Tensordot/GatherV2_1:output:0:model_960/autoencoder/dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0model_960/autoencoder/dense_414/Tensordot/Prod_1?
5model_960/autoencoder/dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5model_960/autoencoder/dense_414/Tensordot/concat/axis?
0model_960/autoencoder/dense_414/Tensordot/concatConcatV27model_960/autoencoder/dense_414/Tensordot/free:output:07model_960/autoencoder/dense_414/Tensordot/axes:output:0>model_960/autoencoder/dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0model_960/autoencoder/dense_414/Tensordot/concat?
/model_960/autoencoder/dense_414/Tensordot/stackPack7model_960/autoencoder/dense_414/Tensordot/Prod:output:09model_960/autoencoder/dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/model_960/autoencoder/dense_414/Tensordot/stack?
3model_960/autoencoder/dense_414/Tensordot/transpose	Transpose(model_960/autoencoder/dense_413/Tanh:y:09model_960/autoencoder/dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????25
3model_960/autoencoder/dense_414/Tensordot/transpose?
1model_960/autoencoder/dense_414/Tensordot/ReshapeReshape7model_960/autoencoder/dense_414/Tensordot/transpose:y:08model_960/autoencoder/dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1model_960/autoencoder/dense_414/Tensordot/Reshape?
0model_960/autoencoder/dense_414/Tensordot/MatMulMatMul:model_960/autoencoder/dense_414/Tensordot/Reshape:output:0@model_960/autoencoder/dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0model_960/autoencoder/dense_414/Tensordot/MatMul?
1model_960/autoencoder/dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?23
1model_960/autoencoder/dense_414/Tensordot/Const_2?
7model_960/autoencoder/dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/autoencoder/dense_414/Tensordot/concat_1/axis?
2model_960/autoencoder/dense_414/Tensordot/concat_1ConcatV2;model_960/autoencoder/dense_414/Tensordot/GatherV2:output:0:model_960/autoencoder/dense_414/Tensordot/Const_2:output:0@model_960/autoencoder/dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2model_960/autoencoder/dense_414/Tensordot/concat_1?
)model_960/autoencoder/dense_414/TensordotReshape:model_960/autoencoder/dense_414/Tensordot/MatMul:product:0;model_960/autoencoder/dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2+
)model_960/autoencoder/dense_414/Tensordot?
6model_960/autoencoder/dense_414/BiasAdd/ReadVariableOpReadVariableOpEmodel_960_autoencoder_dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype028
6model_960/autoencoder/dense_414/BiasAdd/ReadVariableOp?
'model_960/autoencoder/dense_414/BiasAddBiasAdd2model_960/autoencoder/dense_414/Tensordot:output:0>model_960/autoencoder/dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2)
'model_960/autoencoder/dense_414/BiasAdd?
(model_960/discriminator/dropout/IdentityIdentity0model_960/autoencoder/dense_414/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2*
(model_960/discriminator/dropout/Identity?
:model_960/discriminator/dense_415/Tensordot/ReadVariableOpReadVariableOpKmodel_960_discriminator_dense_415_tensordot_readvariableop_dense_415_kernel* 
_output_shapes
:
??*
dtype02<
:model_960/discriminator/dense_415/Tensordot/ReadVariableOp?
0model_960/discriminator/dense_415/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_960/discriminator/dense_415/Tensordot/axes?
0model_960/discriminator/dense_415/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_960/discriminator/dense_415/Tensordot/free?
1model_960/discriminator/dense_415/Tensordot/ShapeShape1model_960/discriminator/dropout/Identity:output:0*
T0*
_output_shapes
:23
1model_960/discriminator/dense_415/Tensordot/Shape?
9model_960/discriminator/dense_415/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_415/Tensordot/GatherV2/axis?
4model_960/discriminator/dense_415/Tensordot/GatherV2GatherV2:model_960/discriminator/dense_415/Tensordot/Shape:output:09model_960/discriminator/dense_415/Tensordot/free:output:0Bmodel_960/discriminator/dense_415/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_960/discriminator/dense_415/Tensordot/GatherV2?
;model_960/discriminator/dense_415/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_960/discriminator/dense_415/Tensordot/GatherV2_1/axis?
6model_960/discriminator/dense_415/Tensordot/GatherV2_1GatherV2:model_960/discriminator/dense_415/Tensordot/Shape:output:09model_960/discriminator/dense_415/Tensordot/axes:output:0Dmodel_960/discriminator/dense_415/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_960/discriminator/dense_415/Tensordot/GatherV2_1?
1model_960/discriminator/dense_415/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/discriminator/dense_415/Tensordot/Const?
0model_960/discriminator/dense_415/Tensordot/ProdProd=model_960/discriminator/dense_415/Tensordot/GatherV2:output:0:model_960/discriminator/dense_415/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_960/discriminator/dense_415/Tensordot/Prod?
3model_960/discriminator/dense_415/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_960/discriminator/dense_415/Tensordot/Const_1?
2model_960/discriminator/dense_415/Tensordot/Prod_1Prod?model_960/discriminator/dense_415/Tensordot/GatherV2_1:output:0<model_960/discriminator/dense_415/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_960/discriminator/dense_415/Tensordot/Prod_1?
7model_960/discriminator/dense_415/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/discriminator/dense_415/Tensordot/concat/axis?
2model_960/discriminator/dense_415/Tensordot/concatConcatV29model_960/discriminator/dense_415/Tensordot/free:output:09model_960/discriminator/dense_415/Tensordot/axes:output:0@model_960/discriminator/dense_415/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_960/discriminator/dense_415/Tensordot/concat?
1model_960/discriminator/dense_415/Tensordot/stackPack9model_960/discriminator/dense_415/Tensordot/Prod:output:0;model_960/discriminator/dense_415/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_960/discriminator/dense_415/Tensordot/stack?
5model_960/discriminator/dense_415/Tensordot/transpose	Transpose1model_960/discriminator/dropout/Identity:output:0;model_960/discriminator/dense_415/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_960/discriminator/dense_415/Tensordot/transpose?
3model_960/discriminator/dense_415/Tensordot/ReshapeReshape9model_960/discriminator/dense_415/Tensordot/transpose:y:0:model_960/discriminator/dense_415/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_960/discriminator/dense_415/Tensordot/Reshape?
2model_960/discriminator/dense_415/Tensordot/MatMulMatMul<model_960/discriminator/dense_415/Tensordot/Reshape:output:0Bmodel_960/discriminator/dense_415/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2model_960/discriminator/dense_415/Tensordot/MatMul?
3model_960/discriminator/dense_415/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?25
3model_960/discriminator/dense_415/Tensordot/Const_2?
9model_960/discriminator/dense_415/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_415/Tensordot/concat_1/axis?
4model_960/discriminator/dense_415/Tensordot/concat_1ConcatV2=model_960/discriminator/dense_415/Tensordot/GatherV2:output:0<model_960/discriminator/dense_415/Tensordot/Const_2:output:0Bmodel_960/discriminator/dense_415/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_960/discriminator/dense_415/Tensordot/concat_1?
+model_960/discriminator/dense_415/TensordotReshape<model_960/discriminator/dense_415/Tensordot/MatMul:product:0=model_960/discriminator/dense_415/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2-
+model_960/discriminator/dense_415/Tensordot?
8model_960/discriminator/dense_415/BiasAdd/ReadVariableOpReadVariableOpGmodel_960_discriminator_dense_415_biasadd_readvariableop_dense_415_bias*
_output_shapes	
:?*
dtype02:
8model_960/discriminator/dense_415/BiasAdd/ReadVariableOp?
)model_960/discriminator/dense_415/BiasAddBiasAdd4model_960/discriminator/dense_415/Tensordot:output:0@model_960/discriminator/dense_415/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2+
)model_960/discriminator/dense_415/BiasAdd?
&model_960/discriminator/dense_415/TanhTanh2model_960/discriminator/dense_415/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2(
&model_960/discriminator/dense_415/Tanh?
:model_960/discriminator/dense_416/Tensordot/ReadVariableOpReadVariableOpKmodel_960_discriminator_dense_416_tensordot_readvariableop_dense_416_kernel* 
_output_shapes
:
??*
dtype02<
:model_960/discriminator/dense_416/Tensordot/ReadVariableOp?
0model_960/discriminator/dense_416/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_960/discriminator/dense_416/Tensordot/axes?
0model_960/discriminator/dense_416/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_960/discriminator/dense_416/Tensordot/free?
1model_960/discriminator/dense_416/Tensordot/ShapeShape*model_960/discriminator/dense_415/Tanh:y:0*
T0*
_output_shapes
:23
1model_960/discriminator/dense_416/Tensordot/Shape?
9model_960/discriminator/dense_416/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_416/Tensordot/GatherV2/axis?
4model_960/discriminator/dense_416/Tensordot/GatherV2GatherV2:model_960/discriminator/dense_416/Tensordot/Shape:output:09model_960/discriminator/dense_416/Tensordot/free:output:0Bmodel_960/discriminator/dense_416/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_960/discriminator/dense_416/Tensordot/GatherV2?
;model_960/discriminator/dense_416/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_960/discriminator/dense_416/Tensordot/GatherV2_1/axis?
6model_960/discriminator/dense_416/Tensordot/GatherV2_1GatherV2:model_960/discriminator/dense_416/Tensordot/Shape:output:09model_960/discriminator/dense_416/Tensordot/axes:output:0Dmodel_960/discriminator/dense_416/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_960/discriminator/dense_416/Tensordot/GatherV2_1?
1model_960/discriminator/dense_416/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/discriminator/dense_416/Tensordot/Const?
0model_960/discriminator/dense_416/Tensordot/ProdProd=model_960/discriminator/dense_416/Tensordot/GatherV2:output:0:model_960/discriminator/dense_416/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_960/discriminator/dense_416/Tensordot/Prod?
3model_960/discriminator/dense_416/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_960/discriminator/dense_416/Tensordot/Const_1?
2model_960/discriminator/dense_416/Tensordot/Prod_1Prod?model_960/discriminator/dense_416/Tensordot/GatherV2_1:output:0<model_960/discriminator/dense_416/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_960/discriminator/dense_416/Tensordot/Prod_1?
7model_960/discriminator/dense_416/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/discriminator/dense_416/Tensordot/concat/axis?
2model_960/discriminator/dense_416/Tensordot/concatConcatV29model_960/discriminator/dense_416/Tensordot/free:output:09model_960/discriminator/dense_416/Tensordot/axes:output:0@model_960/discriminator/dense_416/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_960/discriminator/dense_416/Tensordot/concat?
1model_960/discriminator/dense_416/Tensordot/stackPack9model_960/discriminator/dense_416/Tensordot/Prod:output:0;model_960/discriminator/dense_416/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_960/discriminator/dense_416/Tensordot/stack?
5model_960/discriminator/dense_416/Tensordot/transpose	Transpose*model_960/discriminator/dense_415/Tanh:y:0;model_960/discriminator/dense_416/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_960/discriminator/dense_416/Tensordot/transpose?
3model_960/discriminator/dense_416/Tensordot/ReshapeReshape9model_960/discriminator/dense_416/Tensordot/transpose:y:0:model_960/discriminator/dense_416/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_960/discriminator/dense_416/Tensordot/Reshape?
2model_960/discriminator/dense_416/Tensordot/MatMulMatMul<model_960/discriminator/dense_416/Tensordot/Reshape:output:0Bmodel_960/discriminator/dense_416/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2model_960/discriminator/dense_416/Tensordot/MatMul?
3model_960/discriminator/dense_416/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?25
3model_960/discriminator/dense_416/Tensordot/Const_2?
9model_960/discriminator/dense_416/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_416/Tensordot/concat_1/axis?
4model_960/discriminator/dense_416/Tensordot/concat_1ConcatV2=model_960/discriminator/dense_416/Tensordot/GatherV2:output:0<model_960/discriminator/dense_416/Tensordot/Const_2:output:0Bmodel_960/discriminator/dense_416/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_960/discriminator/dense_416/Tensordot/concat_1?
+model_960/discriminator/dense_416/TensordotReshape<model_960/discriminator/dense_416/Tensordot/MatMul:product:0=model_960/discriminator/dense_416/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2-
+model_960/discriminator/dense_416/Tensordot?
8model_960/discriminator/dense_416/BiasAdd/ReadVariableOpReadVariableOpGmodel_960_discriminator_dense_416_biasadd_readvariableop_dense_416_bias*
_output_shapes	
:?*
dtype02:
8model_960/discriminator/dense_416/BiasAdd/ReadVariableOp?
)model_960/discriminator/dense_416/BiasAddBiasAdd4model_960/discriminator/dense_416/Tensordot:output:0@model_960/discriminator/dense_416/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2+
)model_960/discriminator/dense_416/BiasAdd?
&model_960/discriminator/dense_416/TanhTanh2model_960/discriminator/dense_416/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2(
&model_960/discriminator/dense_416/Tanh?
:model_960/discriminator/dense_417/Tensordot/ReadVariableOpReadVariableOpKmodel_960_discriminator_dense_417_tensordot_readvariableop_dense_417_kernel* 
_output_shapes
:
??*
dtype02<
:model_960/discriminator/dense_417/Tensordot/ReadVariableOp?
0model_960/discriminator/dense_417/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_960/discriminator/dense_417/Tensordot/axes?
0model_960/discriminator/dense_417/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_960/discriminator/dense_417/Tensordot/free?
1model_960/discriminator/dense_417/Tensordot/ShapeShape*model_960/discriminator/dense_416/Tanh:y:0*
T0*
_output_shapes
:23
1model_960/discriminator/dense_417/Tensordot/Shape?
9model_960/discriminator/dense_417/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_417/Tensordot/GatherV2/axis?
4model_960/discriminator/dense_417/Tensordot/GatherV2GatherV2:model_960/discriminator/dense_417/Tensordot/Shape:output:09model_960/discriminator/dense_417/Tensordot/free:output:0Bmodel_960/discriminator/dense_417/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_960/discriminator/dense_417/Tensordot/GatherV2?
;model_960/discriminator/dense_417/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_960/discriminator/dense_417/Tensordot/GatherV2_1/axis?
6model_960/discriminator/dense_417/Tensordot/GatherV2_1GatherV2:model_960/discriminator/dense_417/Tensordot/Shape:output:09model_960/discriminator/dense_417/Tensordot/axes:output:0Dmodel_960/discriminator/dense_417/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_960/discriminator/dense_417/Tensordot/GatherV2_1?
1model_960/discriminator/dense_417/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/discriminator/dense_417/Tensordot/Const?
0model_960/discriminator/dense_417/Tensordot/ProdProd=model_960/discriminator/dense_417/Tensordot/GatherV2:output:0:model_960/discriminator/dense_417/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_960/discriminator/dense_417/Tensordot/Prod?
3model_960/discriminator/dense_417/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_960/discriminator/dense_417/Tensordot/Const_1?
2model_960/discriminator/dense_417/Tensordot/Prod_1Prod?model_960/discriminator/dense_417/Tensordot/GatherV2_1:output:0<model_960/discriminator/dense_417/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_960/discriminator/dense_417/Tensordot/Prod_1?
7model_960/discriminator/dense_417/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/discriminator/dense_417/Tensordot/concat/axis?
2model_960/discriminator/dense_417/Tensordot/concatConcatV29model_960/discriminator/dense_417/Tensordot/free:output:09model_960/discriminator/dense_417/Tensordot/axes:output:0@model_960/discriminator/dense_417/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_960/discriminator/dense_417/Tensordot/concat?
1model_960/discriminator/dense_417/Tensordot/stackPack9model_960/discriminator/dense_417/Tensordot/Prod:output:0;model_960/discriminator/dense_417/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_960/discriminator/dense_417/Tensordot/stack?
5model_960/discriminator/dense_417/Tensordot/transpose	Transpose*model_960/discriminator/dense_416/Tanh:y:0;model_960/discriminator/dense_417/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_960/discriminator/dense_417/Tensordot/transpose?
3model_960/discriminator/dense_417/Tensordot/ReshapeReshape9model_960/discriminator/dense_417/Tensordot/transpose:y:0:model_960/discriminator/dense_417/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_960/discriminator/dense_417/Tensordot/Reshape?
2model_960/discriminator/dense_417/Tensordot/MatMulMatMul<model_960/discriminator/dense_417/Tensordot/Reshape:output:0Bmodel_960/discriminator/dense_417/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2model_960/discriminator/dense_417/Tensordot/MatMul?
3model_960/discriminator/dense_417/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?25
3model_960/discriminator/dense_417/Tensordot/Const_2?
9model_960/discriminator/dense_417/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_417/Tensordot/concat_1/axis?
4model_960/discriminator/dense_417/Tensordot/concat_1ConcatV2=model_960/discriminator/dense_417/Tensordot/GatherV2:output:0<model_960/discriminator/dense_417/Tensordot/Const_2:output:0Bmodel_960/discriminator/dense_417/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_960/discriminator/dense_417/Tensordot/concat_1?
+model_960/discriminator/dense_417/TensordotReshape<model_960/discriminator/dense_417/Tensordot/MatMul:product:0=model_960/discriminator/dense_417/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2-
+model_960/discriminator/dense_417/Tensordot?
8model_960/discriminator/dense_417/BiasAdd/ReadVariableOpReadVariableOpGmodel_960_discriminator_dense_417_biasadd_readvariableop_dense_417_bias*
_output_shapes	
:?*
dtype02:
8model_960/discriminator/dense_417/BiasAdd/ReadVariableOp?
)model_960/discriminator/dense_417/BiasAddBiasAdd4model_960/discriminator/dense_417/Tensordot:output:0@model_960/discriminator/dense_417/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2+
)model_960/discriminator/dense_417/BiasAdd?
&model_960/discriminator/dense_417/TanhTanh2model_960/discriminator/dense_417/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2(
&model_960/discriminator/dense_417/Tanh?
:model_960/discriminator/dense_418/Tensordot/ReadVariableOpReadVariableOpKmodel_960_discriminator_dense_418_tensordot_readvariableop_dense_418_kernel* 
_output_shapes
:
??*
dtype02<
:model_960/discriminator/dense_418/Tensordot/ReadVariableOp?
0model_960/discriminator/dense_418/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_960/discriminator/dense_418/Tensordot/axes?
0model_960/discriminator/dense_418/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_960/discriminator/dense_418/Tensordot/free?
1model_960/discriminator/dense_418/Tensordot/ShapeShape*model_960/discriminator/dense_417/Tanh:y:0*
T0*
_output_shapes
:23
1model_960/discriminator/dense_418/Tensordot/Shape?
9model_960/discriminator/dense_418/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_418/Tensordot/GatherV2/axis?
4model_960/discriminator/dense_418/Tensordot/GatherV2GatherV2:model_960/discriminator/dense_418/Tensordot/Shape:output:09model_960/discriminator/dense_418/Tensordot/free:output:0Bmodel_960/discriminator/dense_418/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_960/discriminator/dense_418/Tensordot/GatherV2?
;model_960/discriminator/dense_418/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_960/discriminator/dense_418/Tensordot/GatherV2_1/axis?
6model_960/discriminator/dense_418/Tensordot/GatherV2_1GatherV2:model_960/discriminator/dense_418/Tensordot/Shape:output:09model_960/discriminator/dense_418/Tensordot/axes:output:0Dmodel_960/discriminator/dense_418/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_960/discriminator/dense_418/Tensordot/GatherV2_1?
1model_960/discriminator/dense_418/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/discriminator/dense_418/Tensordot/Const?
0model_960/discriminator/dense_418/Tensordot/ProdProd=model_960/discriminator/dense_418/Tensordot/GatherV2:output:0:model_960/discriminator/dense_418/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_960/discriminator/dense_418/Tensordot/Prod?
3model_960/discriminator/dense_418/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_960/discriminator/dense_418/Tensordot/Const_1?
2model_960/discriminator/dense_418/Tensordot/Prod_1Prod?model_960/discriminator/dense_418/Tensordot/GatherV2_1:output:0<model_960/discriminator/dense_418/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_960/discriminator/dense_418/Tensordot/Prod_1?
7model_960/discriminator/dense_418/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/discriminator/dense_418/Tensordot/concat/axis?
2model_960/discriminator/dense_418/Tensordot/concatConcatV29model_960/discriminator/dense_418/Tensordot/free:output:09model_960/discriminator/dense_418/Tensordot/axes:output:0@model_960/discriminator/dense_418/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_960/discriminator/dense_418/Tensordot/concat?
1model_960/discriminator/dense_418/Tensordot/stackPack9model_960/discriminator/dense_418/Tensordot/Prod:output:0;model_960/discriminator/dense_418/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_960/discriminator/dense_418/Tensordot/stack?
5model_960/discriminator/dense_418/Tensordot/transpose	Transpose*model_960/discriminator/dense_417/Tanh:y:0;model_960/discriminator/dense_418/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_960/discriminator/dense_418/Tensordot/transpose?
3model_960/discriminator/dense_418/Tensordot/ReshapeReshape9model_960/discriminator/dense_418/Tensordot/transpose:y:0:model_960/discriminator/dense_418/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_960/discriminator/dense_418/Tensordot/Reshape?
2model_960/discriminator/dense_418/Tensordot/MatMulMatMul<model_960/discriminator/dense_418/Tensordot/Reshape:output:0Bmodel_960/discriminator/dense_418/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2model_960/discriminator/dense_418/Tensordot/MatMul?
3model_960/discriminator/dense_418/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?25
3model_960/discriminator/dense_418/Tensordot/Const_2?
9model_960/discriminator/dense_418/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_418/Tensordot/concat_1/axis?
4model_960/discriminator/dense_418/Tensordot/concat_1ConcatV2=model_960/discriminator/dense_418/Tensordot/GatherV2:output:0<model_960/discriminator/dense_418/Tensordot/Const_2:output:0Bmodel_960/discriminator/dense_418/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_960/discriminator/dense_418/Tensordot/concat_1?
+model_960/discriminator/dense_418/TensordotReshape<model_960/discriminator/dense_418/Tensordot/MatMul:product:0=model_960/discriminator/dense_418/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2-
+model_960/discriminator/dense_418/Tensordot?
8model_960/discriminator/dense_418/BiasAdd/ReadVariableOpReadVariableOpGmodel_960_discriminator_dense_418_biasadd_readvariableop_dense_418_bias*
_output_shapes	
:?*
dtype02:
8model_960/discriminator/dense_418/BiasAdd/ReadVariableOp?
)model_960/discriminator/dense_418/BiasAddBiasAdd4model_960/discriminator/dense_418/Tensordot:output:0@model_960/discriminator/dense_418/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2+
)model_960/discriminator/dense_418/BiasAdd?
&model_960/discriminator/dense_418/TanhTanh2model_960/discriminator/dense_418/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2(
&model_960/discriminator/dense_418/Tanh?
:model_960/discriminator/dense_419/Tensordot/ReadVariableOpReadVariableOpKmodel_960_discriminator_dense_419_tensordot_readvariableop_dense_419_kernel*
_output_shapes
:	?*
dtype02<
:model_960/discriminator/dense_419/Tensordot/ReadVariableOp?
0model_960/discriminator/dense_419/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0model_960/discriminator/dense_419/Tensordot/axes?
0model_960/discriminator/dense_419/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_960/discriminator/dense_419/Tensordot/free?
1model_960/discriminator/dense_419/Tensordot/ShapeShape*model_960/discriminator/dense_418/Tanh:y:0*
T0*
_output_shapes
:23
1model_960/discriminator/dense_419/Tensordot/Shape?
9model_960/discriminator/dense_419/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_419/Tensordot/GatherV2/axis?
4model_960/discriminator/dense_419/Tensordot/GatherV2GatherV2:model_960/discriminator/dense_419/Tensordot/Shape:output:09model_960/discriminator/dense_419/Tensordot/free:output:0Bmodel_960/discriminator/dense_419/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4model_960/discriminator/dense_419/Tensordot/GatherV2?
;model_960/discriminator/dense_419/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;model_960/discriminator/dense_419/Tensordot/GatherV2_1/axis?
6model_960/discriminator/dense_419/Tensordot/GatherV2_1GatherV2:model_960/discriminator/dense_419/Tensordot/Shape:output:09model_960/discriminator/dense_419/Tensordot/axes:output:0Dmodel_960/discriminator/dense_419/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6model_960/discriminator/dense_419/Tensordot/GatherV2_1?
1model_960/discriminator/dense_419/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_960/discriminator/dense_419/Tensordot/Const?
0model_960/discriminator/dense_419/Tensordot/ProdProd=model_960/discriminator/dense_419/Tensordot/GatherV2:output:0:model_960/discriminator/dense_419/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0model_960/discriminator/dense_419/Tensordot/Prod?
3model_960/discriminator/dense_419/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3model_960/discriminator/dense_419/Tensordot/Const_1?
2model_960/discriminator/dense_419/Tensordot/Prod_1Prod?model_960/discriminator/dense_419/Tensordot/GatherV2_1:output:0<model_960/discriminator/dense_419/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2model_960/discriminator/dense_419/Tensordot/Prod_1?
7model_960/discriminator/dense_419/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7model_960/discriminator/dense_419/Tensordot/concat/axis?
2model_960/discriminator/dense_419/Tensordot/concatConcatV29model_960/discriminator/dense_419/Tensordot/free:output:09model_960/discriminator/dense_419/Tensordot/axes:output:0@model_960/discriminator/dense_419/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2model_960/discriminator/dense_419/Tensordot/concat?
1model_960/discriminator/dense_419/Tensordot/stackPack9model_960/discriminator/dense_419/Tensordot/Prod:output:0;model_960/discriminator/dense_419/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1model_960/discriminator/dense_419/Tensordot/stack?
5model_960/discriminator/dense_419/Tensordot/transpose	Transpose*model_960/discriminator/dense_418/Tanh:y:0;model_960/discriminator/dense_419/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????27
5model_960/discriminator/dense_419/Tensordot/transpose?
3model_960/discriminator/dense_419/Tensordot/ReshapeReshape9model_960/discriminator/dense_419/Tensordot/transpose:y:0:model_960/discriminator/dense_419/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????25
3model_960/discriminator/dense_419/Tensordot/Reshape?
2model_960/discriminator/dense_419/Tensordot/MatMulMatMul<model_960/discriminator/dense_419/Tensordot/Reshape:output:0Bmodel_960/discriminator/dense_419/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
2model_960/discriminator/dense_419/Tensordot/MatMul?
3model_960/discriminator/dense_419/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_960/discriminator/dense_419/Tensordot/Const_2?
9model_960/discriminator/dense_419/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_960/discriminator/dense_419/Tensordot/concat_1/axis?
4model_960/discriminator/dense_419/Tensordot/concat_1ConcatV2=model_960/discriminator/dense_419/Tensordot/GatherV2:output:0<model_960/discriminator/dense_419/Tensordot/Const_2:output:0Bmodel_960/discriminator/dense_419/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4model_960/discriminator/dense_419/Tensordot/concat_1?
+model_960/discriminator/dense_419/TensordotReshape<model_960/discriminator/dense_419/Tensordot/MatMul:product:0=model_960/discriminator/dense_419/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2-
+model_960/discriminator/dense_419/Tensordot?
8model_960/discriminator/dense_419/BiasAdd/ReadVariableOpReadVariableOpGmodel_960_discriminator_dense_419_biasadd_readvariableop_dense_419_bias*
_output_shapes
:*
dtype02:
8model_960/discriminator/dense_419/BiasAdd/ReadVariableOp?
)model_960/discriminator/dense_419/BiasAddBiasAdd4model_960/discriminator/dense_419/Tensordot:output:0@model_960/discriminator/dense_419/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2+
)model_960/discriminator/dense_419/BiasAdd?
&model_960/discriminator/dense_419/TanhTanh2model_960/discriminator/dense_419/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2(
&model_960/discriminator/dense_419/Tanh?
(model_960/discriminator/flatten_51/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2*
(model_960/discriminator/flatten_51/Const?
*model_960/discriminator/flatten_51/ReshapeReshape*model_960/discriminator/dense_419/Tanh:y:01model_960/discriminator/flatten_51/Const:output:0*
T0*'
_output_shapes
:?????????2,
*model_960/discriminator/flatten_51/Reshape?
7model_960/discriminator/dense_420/MatMul/ReadVariableOpReadVariableOpHmodel_960_discriminator_dense_420_matmul_readvariableop_dense_420_kernel*
_output_shapes

:*
dtype029
7model_960/discriminator/dense_420/MatMul/ReadVariableOp?
(model_960/discriminator/dense_420/MatMulMatMul3model_960/discriminator/flatten_51/Reshape:output:0?model_960/discriminator/dense_420/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(model_960/discriminator/dense_420/MatMul?
8model_960/discriminator/dense_420/BiasAdd/ReadVariableOpReadVariableOpGmodel_960_discriminator_dense_420_biasadd_readvariableop_dense_420_bias*
_output_shapes
:*
dtype02:
8model_960/discriminator/dense_420/BiasAdd/ReadVariableOp?
)model_960/discriminator/dense_420/BiasAddBiasAdd2model_960/discriminator/dense_420/MatMul:product:0@model_960/discriminator/dense_420/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)model_960/discriminator/dense_420/BiasAdd?
)model_960/discriminator/dense_420/SigmoidSigmoid2model_960/discriminator/dense_420/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2+
)model_960/discriminator/dense_420/Sigmoid?	
IdentityIdentity-model_960/discriminator/dense_420/Sigmoid:y:07^model_960/autoencoder/dense_412/BiasAdd/ReadVariableOp9^model_960/autoencoder/dense_412/Tensordot/ReadVariableOp7^model_960/autoencoder/dense_413/BiasAdd/ReadVariableOp9^model_960/autoencoder/dense_413/Tensordot/ReadVariableOp7^model_960/autoencoder/dense_414/BiasAdd/ReadVariableOp9^model_960/autoencoder/dense_414/Tensordot/ReadVariableOp9^model_960/discriminator/dense_415/BiasAdd/ReadVariableOp;^model_960/discriminator/dense_415/Tensordot/ReadVariableOp9^model_960/discriminator/dense_416/BiasAdd/ReadVariableOp;^model_960/discriminator/dense_416/Tensordot/ReadVariableOp9^model_960/discriminator/dense_417/BiasAdd/ReadVariableOp;^model_960/discriminator/dense_417/Tensordot/ReadVariableOp9^model_960/discriminator/dense_418/BiasAdd/ReadVariableOp;^model_960/discriminator/dense_418/Tensordot/ReadVariableOp9^model_960/discriminator/dense_419/BiasAdd/ReadVariableOp;^model_960/discriminator/dense_419/Tensordot/ReadVariableOp9^model_960/discriminator/dense_420/BiasAdd/ReadVariableOp8^model_960/discriminator/dense_420/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::2p
6model_960/autoencoder/dense_412/BiasAdd/ReadVariableOp6model_960/autoencoder/dense_412/BiasAdd/ReadVariableOp2t
8model_960/autoencoder/dense_412/Tensordot/ReadVariableOp8model_960/autoencoder/dense_412/Tensordot/ReadVariableOp2p
6model_960/autoencoder/dense_413/BiasAdd/ReadVariableOp6model_960/autoencoder/dense_413/BiasAdd/ReadVariableOp2t
8model_960/autoencoder/dense_413/Tensordot/ReadVariableOp8model_960/autoencoder/dense_413/Tensordot/ReadVariableOp2p
6model_960/autoencoder/dense_414/BiasAdd/ReadVariableOp6model_960/autoencoder/dense_414/BiasAdd/ReadVariableOp2t
8model_960/autoencoder/dense_414/Tensordot/ReadVariableOp8model_960/autoencoder/dense_414/Tensordot/ReadVariableOp2t
8model_960/discriminator/dense_415/BiasAdd/ReadVariableOp8model_960/discriminator/dense_415/BiasAdd/ReadVariableOp2x
:model_960/discriminator/dense_415/Tensordot/ReadVariableOp:model_960/discriminator/dense_415/Tensordot/ReadVariableOp2t
8model_960/discriminator/dense_416/BiasAdd/ReadVariableOp8model_960/discriminator/dense_416/BiasAdd/ReadVariableOp2x
:model_960/discriminator/dense_416/Tensordot/ReadVariableOp:model_960/discriminator/dense_416/Tensordot/ReadVariableOp2t
8model_960/discriminator/dense_417/BiasAdd/ReadVariableOp8model_960/discriminator/dense_417/BiasAdd/ReadVariableOp2x
:model_960/discriminator/dense_417/Tensordot/ReadVariableOp:model_960/discriminator/dense_417/Tensordot/ReadVariableOp2t
8model_960/discriminator/dense_418/BiasAdd/ReadVariableOp8model_960/discriminator/dense_418/BiasAdd/ReadVariableOp2x
:model_960/discriminator/dense_418/Tensordot/ReadVariableOp:model_960/discriminator/dense_418/Tensordot/ReadVariableOp2t
8model_960/discriminator/dense_419/BiasAdd/ReadVariableOp8model_960/discriminator/dense_419/BiasAdd/ReadVariableOp2x
:model_960/discriminator/dense_419/Tensordot/ReadVariableOp:model_960/discriminator/dense_419/Tensordot/ReadVariableOp2t
8model_960/discriminator/dense_420/BiasAdd/ReadVariableOp8model_960/discriminator/dense_420/BiasAdd/ReadVariableOp2r
7model_960/discriminator/dense_420/MatMul/ReadVariableOp7model_960/discriminator/dense_420/MatMul/ReadVariableOp:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
?
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988405
input_11
dense_412_dense_412_kernel
dense_412_dense_412_bias
dense_413_dense_413_kernel
dense_413_dense_413_bias
dense_414_dense_414_kernel
dense_414_dense_414_bias
identity??!dense_412/StatefulPartitionedCall?!dense_413/StatefulPartitionedCall?!dense_414/StatefulPartitionedCall?
!dense_412/StatefulPartitionedCallStatefulPartitionedCallinput_11dense_412_dense_412_kerneldense_412_dense_412_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_412_layer_call_and_return_conditional_losses_609883072#
!dense_412/StatefulPartitionedCall?
!dense_413/StatefulPartitionedCallStatefulPartitionedCall*dense_412/StatefulPartitionedCall:output:0dense_413_dense_413_kerneldense_413_dense_413_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_413_layer_call_and_return_conditional_losses_609883502#
!dense_413/StatefulPartitionedCall?
!dense_414/StatefulPartitionedCallStatefulPartitionedCall*dense_413/StatefulPartitionedCall:output:0dense_414_dense_414_kerneldense_414_dense_414_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_414_layer_call_and_return_conditional_losses_609883922#
!dense_414/StatefulPartitionedCall?
IdentityIdentity*dense_414/StatefulPartitionedCall:output:0"^dense_412/StatefulPartitionedCall"^dense_413/StatefulPartitionedCall"^dense_414/StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2F
!dense_412/StatefulPartitionedCall!dense_412/StatefulPartitionedCall2F
!dense_413/StatefulPartitionedCall!dense_413/StatefulPartitionedCall2F
!dense_414/StatefulPartitionedCall!dense_414/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
? 
?
G__inference_dense_419_layer_call_and_return_conditional_losses_60988704

inputs-
)tensordot_readvariableop_dense_419_kernel)
%biasadd_readvariableop_dense_419_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_419_kernel*
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
:??????????2
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
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_419_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?(
?
K__inference_discriminator_layer_call_and_return_conditional_losses_60988778
input_12
dense_415_dense_415_kernel
dense_415_dense_415_bias
dense_416_dense_416_kernel
dense_416_dense_416_bias
dense_417_dense_417_kernel
dense_417_dense_417_bias
dense_418_dense_418_kernel
dense_418_dense_418_bias
dense_419_dense_419_kernel
dense_419_dense_419_bias
dense_420_dense_420_kernel
dense_420_dense_420_bias
identity??!dense_415/StatefulPartitionedCall?!dense_416/StatefulPartitionedCall?!dense_417/StatefulPartitionedCall?!dense_418/StatefulPartitionedCall?!dense_419/StatefulPartitionedCall?!dense_420/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallinput_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_609884882
dropout/PartitionedCall?
!dense_415/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_415_dense_415_kerneldense_415_dense_415_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_415_layer_call_and_return_conditional_losses_609885322#
!dense_415/StatefulPartitionedCall?
!dense_416/StatefulPartitionedCallStatefulPartitionedCall*dense_415/StatefulPartitionedCall:output:0dense_416_dense_416_kerneldense_416_dense_416_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_416_layer_call_and_return_conditional_losses_609885752#
!dense_416/StatefulPartitionedCall?
!dense_417/StatefulPartitionedCallStatefulPartitionedCall*dense_416/StatefulPartitionedCall:output:0dense_417_dense_417_kerneldense_417_dense_417_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_417_layer_call_and_return_conditional_losses_609886182#
!dense_417/StatefulPartitionedCall?
!dense_418/StatefulPartitionedCallStatefulPartitionedCall*dense_417/StatefulPartitionedCall:output:0dense_418_dense_418_kerneldense_418_dense_418_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_418_layer_call_and_return_conditional_losses_609886612#
!dense_418/StatefulPartitionedCall?
!dense_419/StatefulPartitionedCallStatefulPartitionedCall*dense_418/StatefulPartitionedCall:output:0dense_419_dense_419_kerneldense_419_dense_419_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_419_layer_call_and_return_conditional_losses_609887042#
!dense_419/StatefulPartitionedCall?
flatten_51/PartitionedCallPartitionedCall*dense_419/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_51_layer_call_and_return_conditional_losses_609887222
flatten_51/PartitionedCall?
!dense_420/StatefulPartitionedCallStatefulPartitionedCall#flatten_51/PartitionedCall:output:0dense_420_dense_420_kerneldense_420_dense_420_bias*
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
G__inference_dense_420_layer_call_and_return_conditional_losses_609887412#
!dense_420/StatefulPartitionedCall?
IdentityIdentity*dense_420/StatefulPartitionedCall:output:0"^dense_415/StatefulPartitionedCall"^dense_416/StatefulPartitionedCall"^dense_417/StatefulPartitionedCall"^dense_418/StatefulPartitionedCall"^dense_419/StatefulPartitionedCall"^dense_420/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::2F
!dense_415/StatefulPartitionedCall!dense_415/StatefulPartitionedCall2F
!dense_416/StatefulPartitionedCall!dense_416/StatefulPartitionedCall2F
!dense_417/StatefulPartitionedCall!dense_417/StatefulPartitionedCall2F
!dense_418/StatefulPartitionedCall!dense_418/StatefulPartitionedCall2F
!dense_419/StatefulPartitionedCall!dense_419/StatefulPartitionedCall2F
!dense_420/StatefulPartitionedCall!dense_420/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_12
?
d
E__inference_dropout_layer_call_and_return_conditional_losses_60990609

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?(
?
K__inference_discriminator_layer_call_and_return_conditional_losses_60988846

inputs
dense_415_dense_415_kernel
dense_415_dense_415_bias
dense_416_dense_416_kernel
dense_416_dense_416_bias
dense_417_dense_417_kernel
dense_417_dense_417_bias
dense_418_dense_418_kernel
dense_418_dense_418_bias
dense_419_dense_419_kernel
dense_419_dense_419_bias
dense_420_dense_420_kernel
dense_420_dense_420_bias
identity??!dense_415/StatefulPartitionedCall?!dense_416/StatefulPartitionedCall?!dense_417/StatefulPartitionedCall?!dense_418/StatefulPartitionedCall?!dense_419/StatefulPartitionedCall?!dense_420/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_609884882
dropout/PartitionedCall?
!dense_415/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_415_dense_415_kerneldense_415_dense_415_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_415_layer_call_and_return_conditional_losses_609885322#
!dense_415/StatefulPartitionedCall?
!dense_416/StatefulPartitionedCallStatefulPartitionedCall*dense_415/StatefulPartitionedCall:output:0dense_416_dense_416_kerneldense_416_dense_416_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_416_layer_call_and_return_conditional_losses_609885752#
!dense_416/StatefulPartitionedCall?
!dense_417/StatefulPartitionedCallStatefulPartitionedCall*dense_416/StatefulPartitionedCall:output:0dense_417_dense_417_kerneldense_417_dense_417_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_417_layer_call_and_return_conditional_losses_609886182#
!dense_417/StatefulPartitionedCall?
!dense_418/StatefulPartitionedCallStatefulPartitionedCall*dense_417/StatefulPartitionedCall:output:0dense_418_dense_418_kerneldense_418_dense_418_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_418_layer_call_and_return_conditional_losses_609886612#
!dense_418/StatefulPartitionedCall?
!dense_419/StatefulPartitionedCallStatefulPartitionedCall*dense_418/StatefulPartitionedCall:output:0dense_419_dense_419_kerneldense_419_dense_419_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_419_layer_call_and_return_conditional_losses_609887042#
!dense_419/StatefulPartitionedCall?
flatten_51/PartitionedCallPartitionedCall*dense_419/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_flatten_51_layer_call_and_return_conditional_losses_609887222
flatten_51/PartitionedCall?
!dense_420/StatefulPartitionedCallStatefulPartitionedCall#flatten_51/PartitionedCall:output:0dense_420_dense_420_kerneldense_420_dense_420_bias*
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
G__inference_dense_420_layer_call_and_return_conditional_losses_609887412#
!dense_420/StatefulPartitionedCall?
IdentityIdentity*dense_420/StatefulPartitionedCall:output:0"^dense_415/StatefulPartitionedCall"^dense_416/StatefulPartitionedCall"^dense_417/StatefulPartitionedCall"^dense_418/StatefulPartitionedCall"^dense_419/StatefulPartitionedCall"^dense_420/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::2F
!dense_415/StatefulPartitionedCall!dense_415/StatefulPartitionedCall2F
!dense_416/StatefulPartitionedCall!dense_416/StatefulPartitionedCall2F
!dense_417/StatefulPartitionedCall!dense_417/StatefulPartitionedCall2F
!dense_418/StatefulPartitionedCall!dense_418/StatefulPartitionedCall2F
!dense_419/StatefulPartitionedCall!dense_419/StatefulPartitionedCall2F
!dense_420/StatefulPartitionedCall!dense_420/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?q
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60990039
inputs_07
3dense_412_tensordot_readvariableop_dense_412_kernel3
/dense_412_biasadd_readvariableop_dense_412_bias7
3dense_413_tensordot_readvariableop_dense_413_kernel3
/dense_413_biasadd_readvariableop_dense_413_bias7
3dense_414_tensordot_readvariableop_dense_414_kernel3
/dense_414_biasadd_readvariableop_dense_414_bias
identity?? dense_412/BiasAdd/ReadVariableOp?"dense_412/Tensordot/ReadVariableOp? dense_413/BiasAdd/ReadVariableOp?"dense_413/Tensordot/ReadVariableOp? dense_414/BiasAdd/ReadVariableOp?"dense_414/Tensordot/ReadVariableOp?
"dense_412/Tensordot/ReadVariableOpReadVariableOp3dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype02$
"dense_412/Tensordot/ReadVariableOp~
dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_412/Tensordot/axes?
dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_412/Tensordot/freen
dense_412/Tensordot/ShapeShapeinputs_0*
T0*
_output_shapes
:2
dense_412/Tensordot/Shape?
!dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/GatherV2/axis?
dense_412/Tensordot/GatherV2GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/free:output:0*dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_412/Tensordot/GatherV2?
#dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_412/Tensordot/GatherV2_1/axis?
dense_412/Tensordot/GatherV2_1GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/axes:output:0,dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_412/Tensordot/GatherV2_1?
dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const?
dense_412/Tensordot/ProdProd%dense_412/Tensordot/GatherV2:output:0"dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod?
dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_1?
dense_412/Tensordot/Prod_1Prod'dense_412/Tensordot/GatherV2_1:output:0$dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod_1?
dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_412/Tensordot/concat/axis?
dense_412/Tensordot/concatConcatV2!dense_412/Tensordot/free:output:0!dense_412/Tensordot/axes:output:0(dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat?
dense_412/Tensordot/stackPack!dense_412/Tensordot/Prod:output:0#dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/stack?
dense_412/Tensordot/transpose	Transposeinputs_0#dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_412/Tensordot/transpose?
dense_412/Tensordot/ReshapeReshape!dense_412/Tensordot/transpose:y:0"dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_412/Tensordot/Reshape?
dense_412/Tensordot/MatMulMatMul$dense_412/Tensordot/Reshape:output:0*dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_412/Tensordot/MatMul?
dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_2?
!dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/concat_1/axis?
dense_412/Tensordot/concat_1ConcatV2%dense_412/Tensordot/GatherV2:output:0$dense_412/Tensordot/Const_2:output:0*dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat_1?
dense_412/TensordotReshape$dense_412/Tensordot/MatMul:product:0%dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tensordot?
 dense_412/BiasAdd/ReadVariableOpReadVariableOp/dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02"
 dense_412/BiasAdd/ReadVariableOp?
dense_412/BiasAddBiasAdddense_412/Tensordot:output:0(dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_412/BiasAddz
dense_412/TanhTanhdense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tanh?
"dense_413/Tensordot/ReadVariableOpReadVariableOp3dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_413/Tensordot/ReadVariableOp~
dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_413/Tensordot/axes?
dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_413/Tensordot/freex
dense_413/Tensordot/ShapeShapedense_412/Tanh:y:0*
T0*
_output_shapes
:2
dense_413/Tensordot/Shape?
!dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/GatherV2/axis?
dense_413/Tensordot/GatherV2GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/free:output:0*dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_413/Tensordot/GatherV2?
#dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_413/Tensordot/GatherV2_1/axis?
dense_413/Tensordot/GatherV2_1GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/axes:output:0,dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_413/Tensordot/GatherV2_1?
dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const?
dense_413/Tensordot/ProdProd%dense_413/Tensordot/GatherV2:output:0"dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod?
dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const_1?
dense_413/Tensordot/Prod_1Prod'dense_413/Tensordot/GatherV2_1:output:0$dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod_1?
dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_413/Tensordot/concat/axis?
dense_413/Tensordot/concatConcatV2!dense_413/Tensordot/free:output:0!dense_413/Tensordot/axes:output:0(dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat?
dense_413/Tensordot/stackPack!dense_413/Tensordot/Prod:output:0#dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/stack?
dense_413/Tensordot/transpose	Transposedense_412/Tanh:y:0#dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_413/Tensordot/transpose?
dense_413/Tensordot/ReshapeReshape!dense_413/Tensordot/transpose:y:0"dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_413/Tensordot/Reshape?
dense_413/Tensordot/MatMulMatMul$dense_413/Tensordot/Reshape:output:0*dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_413/Tensordot/MatMul?
dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_413/Tensordot/Const_2?
!dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/concat_1/axis?
dense_413/Tensordot/concat_1ConcatV2%dense_413/Tensordot/GatherV2:output:0$dense_413/Tensordot/Const_2:output:0*dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat_1?
dense_413/TensordotReshape$dense_413/Tensordot/MatMul:product:0%dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tensordot?
 dense_413/BiasAdd/ReadVariableOpReadVariableOp/dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02"
 dense_413/BiasAdd/ReadVariableOp?
dense_413/BiasAddBiasAdddense_413/Tensordot:output:0(dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_413/BiasAdd{
dense_413/TanhTanhdense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tanh?
"dense_414/Tensordot/ReadVariableOpReadVariableOp3dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype02$
"dense_414/Tensordot/ReadVariableOp~
dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_414/Tensordot/axes?
dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_414/Tensordot/freex
dense_414/Tensordot/ShapeShapedense_413/Tanh:y:0*
T0*
_output_shapes
:2
dense_414/Tensordot/Shape?
!dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/GatherV2/axis?
dense_414/Tensordot/GatherV2GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/free:output:0*dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_414/Tensordot/GatherV2?
#dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_414/Tensordot/GatherV2_1/axis?
dense_414/Tensordot/GatherV2_1GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/axes:output:0,dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_414/Tensordot/GatherV2_1?
dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const?
dense_414/Tensordot/ProdProd%dense_414/Tensordot/GatherV2:output:0"dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod?
dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const_1?
dense_414/Tensordot/Prod_1Prod'dense_414/Tensordot/GatherV2_1:output:0$dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod_1?
dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_414/Tensordot/concat/axis?
dense_414/Tensordot/concatConcatV2!dense_414/Tensordot/free:output:0!dense_414/Tensordot/axes:output:0(dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat?
dense_414/Tensordot/stackPack!dense_414/Tensordot/Prod:output:0#dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/stack?
dense_414/Tensordot/transpose	Transposedense_413/Tanh:y:0#dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot/transpose?
dense_414/Tensordot/ReshapeReshape!dense_414/Tensordot/transpose:y:0"dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_414/Tensordot/Reshape?
dense_414/Tensordot/MatMulMatMul$dense_414/Tensordot/Reshape:output:0*dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_414/Tensordot/MatMul?
dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_414/Tensordot/Const_2?
!dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/concat_1/axis?
dense_414/Tensordot/concat_1ConcatV2%dense_414/Tensordot/GatherV2:output:0$dense_414/Tensordot/Const_2:output:0*dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat_1?
dense_414/TensordotReshape$dense_414/Tensordot/MatMul:product:0%dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot?
 dense_414/BiasAdd/ReadVariableOpReadVariableOp/dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02"
 dense_414/BiasAdd/ReadVariableOp?
dense_414/BiasAddBiasAdddense_414/Tensordot:output:0(dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_414/BiasAdd?
IdentityIdentitydense_414/BiasAdd:output:0!^dense_412/BiasAdd/ReadVariableOp#^dense_412/Tensordot/ReadVariableOp!^dense_413/BiasAdd/ReadVariableOp#^dense_413/Tensordot/ReadVariableOp!^dense_414/BiasAdd/ReadVariableOp#^dense_414/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_412/BiasAdd/ReadVariableOp dense_412/BiasAdd/ReadVariableOp2H
"dense_412/Tensordot/ReadVariableOp"dense_412/Tensordot/ReadVariableOp2D
 dense_413/BiasAdd/ReadVariableOp dense_413/BiasAdd/ReadVariableOp2H
"dense_413/Tensordot/ReadVariableOp"dense_413/Tensordot/ReadVariableOp2D
 dense_414/BiasAdd/ReadVariableOp dense_414/BiasAdd/ReadVariableOp2H
"dense_414/Tensordot/ReadVariableOp"dense_414/Tensordot/ReadVariableOp:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?
F
*__inference_dropout_layer_call_fn_60990624

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_609884882
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_418_layer_call_fn_60990776

inputs
dense_418_kernel
dense_418_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_418_kerneldense_418_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_418_layer_call_and_return_conditional_losses_609886612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_60989254
input_11
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11dense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_biasdense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_609882722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
?
?
G__inference_model_960_layer_call_and_return_conditional_losses_60989134
input_11 
autoencoder_dense_412_kernel
autoencoder_dense_412_bias 
autoencoder_dense_413_kernel
autoencoder_dense_413_bias 
autoencoder_dense_414_kernel
autoencoder_dense_414_bias"
discriminator_dense_415_kernel 
discriminator_dense_415_bias"
discriminator_dense_416_kernel 
discriminator_dense_416_bias"
discriminator_dense_417_kernel 
discriminator_dense_417_bias"
discriminator_dense_418_kernel 
discriminator_dense_418_bias"
discriminator_dense_419_kernel 
discriminator_dense_419_bias"
discriminator_dense_420_kernel 
discriminator_dense_420_bias
identity??#autoencoder/StatefulPartitionedCall?%discriminator/StatefulPartitionedCall?
#autoencoder/StatefulPartitionedCallStatefulPartitionedCallinput_11autoencoder_dense_412_kernelautoencoder_dense_412_biasautoencoder_dense_413_kernelautoencoder_dense_413_biasautoencoder_dense_414_kernelautoencoder_dense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609890322%
#autoencoder/StatefulPartitionedCall?
%discriminator/StatefulPartitionedCallStatefulPartitionedCall,autoencoder/StatefulPartitionedCall:output:0discriminator_dense_415_kerneldiscriminator_dense_415_biasdiscriminator_dense_416_kerneldiscriminator_dense_416_biasdiscriminator_dense_417_kerneldiscriminator_dense_417_biasdiscriminator_dense_418_kerneldiscriminator_dense_418_biasdiscriminator_dense_419_kerneldiscriminator_dense_419_biasdiscriminator_dense_420_kerneldiscriminator_dense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609888462'
%discriminator/StatefulPartitionedCall?
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0$^autoencoder/StatefulPartitionedCall&^discriminator/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::2J
#autoencoder/StatefulPartitionedCall#autoencoder/StatefulPartitionedCall2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
?
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988458

inputs
dense_412_dense_412_kernel
dense_412_dense_412_bias
dense_413_dense_413_kernel
dense_413_dense_413_bias
dense_414_dense_414_kernel
dense_414_dense_414_bias
identity??!dense_412/StatefulPartitionedCall?!dense_413/StatefulPartitionedCall?!dense_414/StatefulPartitionedCall?
!dense_412/StatefulPartitionedCallStatefulPartitionedCallinputsdense_412_dense_412_kerneldense_412_dense_412_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_412_layer_call_and_return_conditional_losses_609883072#
!dense_412/StatefulPartitionedCall?
!dense_413/StatefulPartitionedCallStatefulPartitionedCall*dense_412/StatefulPartitionedCall:output:0dense_413_dense_413_kerneldense_413_dense_413_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_413_layer_call_and_return_conditional_losses_609883502#
!dense_413/StatefulPartitionedCall?
!dense_414/StatefulPartitionedCallStatefulPartitionedCall*dense_413/StatefulPartitionedCall:output:0dense_414_dense_414_kerneldense_414_dense_414_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_414_layer_call_and_return_conditional_losses_609883922#
!dense_414/StatefulPartitionedCall?
IdentityIdentity*dense_414/StatefulPartitionedCall:output:0"^dense_412/StatefulPartitionedCall"^dense_413/StatefulPartitionedCall"^dense_414/StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2F
!dense_412/StatefulPartitionedCall!dense_412/StatefulPartitionedCall2F
!dense_413/StatefulPartitionedCall!dense_413/StatefulPartitionedCall2F
!dense_414/StatefulPartitionedCall!dense_414/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_416_layer_call_fn_60990700

inputs
dense_416_kernel
dense_416_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_416_kerneldense_416_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_416_layer_call_and_return_conditional_losses_609885752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_60990134
inputs_0
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0dense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609889482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?

?
G__inference_dense_420_layer_call_and_return_conditional_losses_60988741

inputs*
&matmul_readvariableop_dense_420_kernel)
%biasadd_readvariableop_dense_420_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp&matmul_readvariableop_dense_420_kernel*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_420_bias*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_415_layer_call_fn_60990662

inputs
dense_415_kernel
dense_415_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_415_kerneldense_415_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_415_layer_call_and_return_conditional_losses_609885322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
0__inference_discriminator_layer_call_fn_60990484

inputs
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609888462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?!
!__inference__traced_save_60991076
file_prefix5
1savev2_training_318_adam_iter_read_readvariableop	7
3savev2_training_318_adam_beta_1_read_readvariableop7
3savev2_training_318_adam_beta_2_read_readvariableop6
2savev2_training_318_adam_decay_read_readvariableop>
:savev2_training_318_adam_learning_rate_read_readvariableop/
+savev2_dense_412_kernel_read_readvariableop-
)savev2_dense_412_bias_read_readvariableop/
+savev2_dense_413_kernel_read_readvariableop-
)savev2_dense_413_bias_read_readvariableop/
+savev2_dense_414_kernel_read_readvariableop-
)savev2_dense_414_bias_read_readvariableop/
+savev2_dense_415_kernel_read_readvariableop-
)savev2_dense_415_bias_read_readvariableop/
+savev2_dense_416_kernel_read_readvariableop-
)savev2_dense_416_bias_read_readvariableop/
+savev2_dense_417_kernel_read_readvariableop-
)savev2_dense_417_bias_read_readvariableop/
+savev2_dense_418_kernel_read_readvariableop-
)savev2_dense_418_bias_read_readvariableop/
+savev2_dense_419_kernel_read_readvariableop-
)savev2_dense_419_bias_read_readvariableop/
+savev2_dense_420_kernel_read_readvariableop-
)savev2_dense_420_bias_read_readvariableop5
1savev2_training_130_adam_iter_read_readvariableop	7
3savev2_training_130_adam_beta_1_read_readvariableop7
3savev2_training_130_adam_beta_2_read_readvariableop6
2savev2_training_130_adam_decay_read_readvariableop>
:savev2_training_130_adam_learning_rate_read_readvariableop(
$savev2_total_329_read_readvariableop(
$savev2_count_329_read_readvariableop(
$savev2_total_139_read_readvariableop(
$savev2_count_139_read_readvariableop(
$savev2_total_140_read_readvariableop(
$savev2_count_140_read_readvariableopC
?savev2_training_318_adam_dense_412_kernel_m_read_readvariableopA
=savev2_training_318_adam_dense_412_bias_m_read_readvariableopC
?savev2_training_318_adam_dense_413_kernel_m_read_readvariableopA
=savev2_training_318_adam_dense_413_bias_m_read_readvariableopC
?savev2_training_318_adam_dense_414_kernel_m_read_readvariableopA
=savev2_training_318_adam_dense_414_bias_m_read_readvariableopC
?savev2_training_318_adam_dense_412_kernel_v_read_readvariableopA
=savev2_training_318_adam_dense_412_bias_v_read_readvariableopC
?savev2_training_318_adam_dense_413_kernel_v_read_readvariableopA
=savev2_training_318_adam_dense_413_bias_v_read_readvariableopC
?savev2_training_318_adam_dense_414_kernel_v_read_readvariableopA
=savev2_training_318_adam_dense_414_bias_v_read_readvariableopC
?savev2_training_130_adam_dense_415_kernel_m_read_readvariableopA
=savev2_training_130_adam_dense_415_bias_m_read_readvariableopC
?savev2_training_130_adam_dense_416_kernel_m_read_readvariableopA
=savev2_training_130_adam_dense_416_bias_m_read_readvariableopC
?savev2_training_130_adam_dense_417_kernel_m_read_readvariableopA
=savev2_training_130_adam_dense_417_bias_m_read_readvariableopC
?savev2_training_130_adam_dense_418_kernel_m_read_readvariableopA
=savev2_training_130_adam_dense_418_bias_m_read_readvariableopC
?savev2_training_130_adam_dense_419_kernel_m_read_readvariableopA
=savev2_training_130_adam_dense_419_bias_m_read_readvariableopC
?savev2_training_130_adam_dense_420_kernel_m_read_readvariableopA
=savev2_training_130_adam_dense_420_bias_m_read_readvariableopC
?savev2_training_130_adam_dense_415_kernel_v_read_readvariableopA
=savev2_training_130_adam_dense_415_bias_v_read_readvariableopC
?savev2_training_130_adam_dense_416_kernel_v_read_readvariableopA
=savev2_training_130_adam_dense_416_bias_v_read_readvariableopC
?savev2_training_130_adam_dense_417_kernel_v_read_readvariableopA
=savev2_training_130_adam_dense_417_bias_v_read_readvariableopC
?savev2_training_130_adam_dense_418_kernel_v_read_readvariableopA
=savev2_training_130_adam_dense_418_bias_v_read_readvariableopC
?savev2_training_130_adam_dense_419_kernel_v_read_readvariableopA
=savev2_training_130_adam_dense_419_bias_v_read_readvariableopC
?savev2_training_130_adam_dense_420_kernel_v_read_readvariableopA
=savev2_training_130_adam_dense_420_bias_v_read_readvariableop
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
ShardedFilename?'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*?&
value?&B?&GB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/16/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/16/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*?
value?B?GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices? 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_training_318_adam_iter_read_readvariableop3savev2_training_318_adam_beta_1_read_readvariableop3savev2_training_318_adam_beta_2_read_readvariableop2savev2_training_318_adam_decay_read_readvariableop:savev2_training_318_adam_learning_rate_read_readvariableop+savev2_dense_412_kernel_read_readvariableop)savev2_dense_412_bias_read_readvariableop+savev2_dense_413_kernel_read_readvariableop)savev2_dense_413_bias_read_readvariableop+savev2_dense_414_kernel_read_readvariableop)savev2_dense_414_bias_read_readvariableop+savev2_dense_415_kernel_read_readvariableop)savev2_dense_415_bias_read_readvariableop+savev2_dense_416_kernel_read_readvariableop)savev2_dense_416_bias_read_readvariableop+savev2_dense_417_kernel_read_readvariableop)savev2_dense_417_bias_read_readvariableop+savev2_dense_418_kernel_read_readvariableop)savev2_dense_418_bias_read_readvariableop+savev2_dense_419_kernel_read_readvariableop)savev2_dense_419_bias_read_readvariableop+savev2_dense_420_kernel_read_readvariableop)savev2_dense_420_bias_read_readvariableop1savev2_training_130_adam_iter_read_readvariableop3savev2_training_130_adam_beta_1_read_readvariableop3savev2_training_130_adam_beta_2_read_readvariableop2savev2_training_130_adam_decay_read_readvariableop:savev2_training_130_adam_learning_rate_read_readvariableop$savev2_total_329_read_readvariableop$savev2_count_329_read_readvariableop$savev2_total_139_read_readvariableop$savev2_count_139_read_readvariableop$savev2_total_140_read_readvariableop$savev2_count_140_read_readvariableop?savev2_training_318_adam_dense_412_kernel_m_read_readvariableop=savev2_training_318_adam_dense_412_bias_m_read_readvariableop?savev2_training_318_adam_dense_413_kernel_m_read_readvariableop=savev2_training_318_adam_dense_413_bias_m_read_readvariableop?savev2_training_318_adam_dense_414_kernel_m_read_readvariableop=savev2_training_318_adam_dense_414_bias_m_read_readvariableop?savev2_training_318_adam_dense_412_kernel_v_read_readvariableop=savev2_training_318_adam_dense_412_bias_v_read_readvariableop?savev2_training_318_adam_dense_413_kernel_v_read_readvariableop=savev2_training_318_adam_dense_413_bias_v_read_readvariableop?savev2_training_318_adam_dense_414_kernel_v_read_readvariableop=savev2_training_318_adam_dense_414_bias_v_read_readvariableop?savev2_training_130_adam_dense_415_kernel_m_read_readvariableop=savev2_training_130_adam_dense_415_bias_m_read_readvariableop?savev2_training_130_adam_dense_416_kernel_m_read_readvariableop=savev2_training_130_adam_dense_416_bias_m_read_readvariableop?savev2_training_130_adam_dense_417_kernel_m_read_readvariableop=savev2_training_130_adam_dense_417_bias_m_read_readvariableop?savev2_training_130_adam_dense_418_kernel_m_read_readvariableop=savev2_training_130_adam_dense_418_bias_m_read_readvariableop?savev2_training_130_adam_dense_419_kernel_m_read_readvariableop=savev2_training_130_adam_dense_419_bias_m_read_readvariableop?savev2_training_130_adam_dense_420_kernel_m_read_readvariableop=savev2_training_130_adam_dense_420_bias_m_read_readvariableop?savev2_training_130_adam_dense_415_kernel_v_read_readvariableop=savev2_training_130_adam_dense_415_bias_v_read_readvariableop?savev2_training_130_adam_dense_416_kernel_v_read_readvariableop=savev2_training_130_adam_dense_416_bias_v_read_readvariableop?savev2_training_130_adam_dense_417_kernel_v_read_readvariableop=savev2_training_130_adam_dense_417_bias_v_read_readvariableop?savev2_training_130_adam_dense_418_kernel_v_read_readvariableop=savev2_training_130_adam_dense_418_bias_v_read_readvariableop?savev2_training_130_adam_dense_419_kernel_v_read_readvariableop=savev2_training_130_adam_dense_419_bias_v_read_readvariableop?savev2_training_130_adam_dense_420_kernel_v_read_readvariableop=savev2_training_130_adam_dense_420_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *U
dtypesK
I2G		2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	? : :	 ?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?:::: : : : : : : : : : : :	? : :	 ?:?:
??:?:	? : :	 ?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::::
??:?:
??:?:
??:?:
??:?:	?:::: 2(
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
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::
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
: :!

_output_shapes
: :"

_output_shapes
: :%#!

_output_shapes
:	? : $

_output_shapes
: :%%!

_output_shapes
:	 ?:!&

_output_shapes	
:?:&'"
 
_output_shapes
:
??:!(

_output_shapes	
:?:%)!

_output_shapes
:	? : *

_output_shapes
: :%+!

_output_shapes
:	 ?:!,
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
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:&3"
 
_output_shapes
:
??:!4

_output_shapes	
:?:&5"
 
_output_shapes
:
??:!6

_output_shapes	
:?:%7!

_output_shapes
:	?: 8

_output_shapes
::$9 

_output_shapes

:: :

_output_shapes
::&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:&?"
 
_output_shapes
:
??:!@

_output_shapes	
:?:&A"
 
_output_shapes
:
??:!B

_output_shapes	
:?:%C!

_output_shapes
:	?: D

_output_shapes
::$E 

_output_shapes

:: F

_output_shapes
::G

_output_shapes
: 
?p
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988948

inputs7
3dense_412_tensordot_readvariableop_dense_412_kernel3
/dense_412_biasadd_readvariableop_dense_412_bias7
3dense_413_tensordot_readvariableop_dense_413_kernel3
/dense_413_biasadd_readvariableop_dense_413_bias7
3dense_414_tensordot_readvariableop_dense_414_kernel3
/dense_414_biasadd_readvariableop_dense_414_bias
identity?? dense_412/BiasAdd/ReadVariableOp?"dense_412/Tensordot/ReadVariableOp? dense_413/BiasAdd/ReadVariableOp?"dense_413/Tensordot/ReadVariableOp? dense_414/BiasAdd/ReadVariableOp?"dense_414/Tensordot/ReadVariableOp?
"dense_412/Tensordot/ReadVariableOpReadVariableOp3dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype02$
"dense_412/Tensordot/ReadVariableOp~
dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_412/Tensordot/axes?
dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_412/Tensordot/freel
dense_412/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_412/Tensordot/Shape?
!dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/GatherV2/axis?
dense_412/Tensordot/GatherV2GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/free:output:0*dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_412/Tensordot/GatherV2?
#dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_412/Tensordot/GatherV2_1/axis?
dense_412/Tensordot/GatherV2_1GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/axes:output:0,dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_412/Tensordot/GatherV2_1?
dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const?
dense_412/Tensordot/ProdProd%dense_412/Tensordot/GatherV2:output:0"dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod?
dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_1?
dense_412/Tensordot/Prod_1Prod'dense_412/Tensordot/GatherV2_1:output:0$dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod_1?
dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_412/Tensordot/concat/axis?
dense_412/Tensordot/concatConcatV2!dense_412/Tensordot/free:output:0!dense_412/Tensordot/axes:output:0(dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat?
dense_412/Tensordot/stackPack!dense_412/Tensordot/Prod:output:0#dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/stack?
dense_412/Tensordot/transpose	Transposeinputs#dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_412/Tensordot/transpose?
dense_412/Tensordot/ReshapeReshape!dense_412/Tensordot/transpose:y:0"dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_412/Tensordot/Reshape?
dense_412/Tensordot/MatMulMatMul$dense_412/Tensordot/Reshape:output:0*dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_412/Tensordot/MatMul?
dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_2?
!dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/concat_1/axis?
dense_412/Tensordot/concat_1ConcatV2%dense_412/Tensordot/GatherV2:output:0$dense_412/Tensordot/Const_2:output:0*dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat_1?
dense_412/TensordotReshape$dense_412/Tensordot/MatMul:product:0%dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tensordot?
 dense_412/BiasAdd/ReadVariableOpReadVariableOp/dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02"
 dense_412/BiasAdd/ReadVariableOp?
dense_412/BiasAddBiasAdddense_412/Tensordot:output:0(dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_412/BiasAddz
dense_412/TanhTanhdense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tanh?
"dense_413/Tensordot/ReadVariableOpReadVariableOp3dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_413/Tensordot/ReadVariableOp~
dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_413/Tensordot/axes?
dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_413/Tensordot/freex
dense_413/Tensordot/ShapeShapedense_412/Tanh:y:0*
T0*
_output_shapes
:2
dense_413/Tensordot/Shape?
!dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/GatherV2/axis?
dense_413/Tensordot/GatherV2GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/free:output:0*dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_413/Tensordot/GatherV2?
#dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_413/Tensordot/GatherV2_1/axis?
dense_413/Tensordot/GatherV2_1GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/axes:output:0,dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_413/Tensordot/GatherV2_1?
dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const?
dense_413/Tensordot/ProdProd%dense_413/Tensordot/GatherV2:output:0"dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod?
dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const_1?
dense_413/Tensordot/Prod_1Prod'dense_413/Tensordot/GatherV2_1:output:0$dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod_1?
dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_413/Tensordot/concat/axis?
dense_413/Tensordot/concatConcatV2!dense_413/Tensordot/free:output:0!dense_413/Tensordot/axes:output:0(dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat?
dense_413/Tensordot/stackPack!dense_413/Tensordot/Prod:output:0#dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/stack?
dense_413/Tensordot/transpose	Transposedense_412/Tanh:y:0#dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_413/Tensordot/transpose?
dense_413/Tensordot/ReshapeReshape!dense_413/Tensordot/transpose:y:0"dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_413/Tensordot/Reshape?
dense_413/Tensordot/MatMulMatMul$dense_413/Tensordot/Reshape:output:0*dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_413/Tensordot/MatMul?
dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_413/Tensordot/Const_2?
!dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/concat_1/axis?
dense_413/Tensordot/concat_1ConcatV2%dense_413/Tensordot/GatherV2:output:0$dense_413/Tensordot/Const_2:output:0*dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat_1?
dense_413/TensordotReshape$dense_413/Tensordot/MatMul:product:0%dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tensordot?
 dense_413/BiasAdd/ReadVariableOpReadVariableOp/dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02"
 dense_413/BiasAdd/ReadVariableOp?
dense_413/BiasAddBiasAdddense_413/Tensordot:output:0(dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_413/BiasAdd{
dense_413/TanhTanhdense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tanh?
"dense_414/Tensordot/ReadVariableOpReadVariableOp3dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype02$
"dense_414/Tensordot/ReadVariableOp~
dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_414/Tensordot/axes?
dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_414/Tensordot/freex
dense_414/Tensordot/ShapeShapedense_413/Tanh:y:0*
T0*
_output_shapes
:2
dense_414/Tensordot/Shape?
!dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/GatherV2/axis?
dense_414/Tensordot/GatherV2GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/free:output:0*dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_414/Tensordot/GatherV2?
#dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_414/Tensordot/GatherV2_1/axis?
dense_414/Tensordot/GatherV2_1GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/axes:output:0,dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_414/Tensordot/GatherV2_1?
dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const?
dense_414/Tensordot/ProdProd%dense_414/Tensordot/GatherV2:output:0"dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod?
dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const_1?
dense_414/Tensordot/Prod_1Prod'dense_414/Tensordot/GatherV2_1:output:0$dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod_1?
dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_414/Tensordot/concat/axis?
dense_414/Tensordot/concatConcatV2!dense_414/Tensordot/free:output:0!dense_414/Tensordot/axes:output:0(dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat?
dense_414/Tensordot/stackPack!dense_414/Tensordot/Prod:output:0#dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/stack?
dense_414/Tensordot/transpose	Transposedense_413/Tanh:y:0#dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot/transpose?
dense_414/Tensordot/ReshapeReshape!dense_414/Tensordot/transpose:y:0"dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_414/Tensordot/Reshape?
dense_414/Tensordot/MatMulMatMul$dense_414/Tensordot/Reshape:output:0*dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_414/Tensordot/MatMul?
dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_414/Tensordot/Const_2?
!dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/concat_1/axis?
dense_414/Tensordot/concat_1ConcatV2%dense_414/Tensordot/GatherV2:output:0$dense_414/Tensordot/Const_2:output:0*dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat_1?
dense_414/TensordotReshape$dense_414/Tensordot/MatMul:product:0%dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot?
 dense_414/BiasAdd/ReadVariableOpReadVariableOp/dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02"
 dense_414/BiasAdd/ReadVariableOp?
dense_414/BiasAddBiasAdddense_414/Tensordot:output:0(dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_414/BiasAdd?
IdentityIdentitydense_414/BiasAdd:output:0!^dense_412/BiasAdd/ReadVariableOp#^dense_412/Tensordot/ReadVariableOp!^dense_413/BiasAdd/ReadVariableOp#^dense_413/Tensordot/ReadVariableOp!^dense_414/BiasAdd/ReadVariableOp#^dense_414/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_412/BiasAdd/ReadVariableOp dense_412/BiasAdd/ReadVariableOp2H
"dense_412/Tensordot/ReadVariableOp"dense_412/Tensordot/ReadVariableOp2D
 dense_413/BiasAdd/ReadVariableOp dense_413/BiasAdd/ReadVariableOp2H
"dense_413/Tensordot/ReadVariableOp"dense_413/Tensordot/ReadVariableOp2D
 dense_414/BiasAdd/ReadVariableOp dense_414/BiasAdd/ReadVariableOp2H
"dense_414/Tensordot/ReadVariableOp"dense_414/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_layer_call_and_return_conditional_losses_60988488

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_417_layer_call_and_return_conditional_losses_60988618

inputs-
)tensordot_readvariableop_dense_417_kernel)
%biasadd_readvariableop_dense_417_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_417_kernel* 
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_417_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_layer_call_and_return_conditional_losses_60988483

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?	
K__inference_discriminator_layer_call_and_return_conditional_losses_60990301

inputs7
3dense_415_tensordot_readvariableop_dense_415_kernel3
/dense_415_biasadd_readvariableop_dense_415_bias7
3dense_416_tensordot_readvariableop_dense_416_kernel3
/dense_416_biasadd_readvariableop_dense_416_bias7
3dense_417_tensordot_readvariableop_dense_417_kernel3
/dense_417_biasadd_readvariableop_dense_417_bias7
3dense_418_tensordot_readvariableop_dense_418_kernel3
/dense_418_biasadd_readvariableop_dense_418_bias7
3dense_419_tensordot_readvariableop_dense_419_kernel3
/dense_419_biasadd_readvariableop_dense_419_bias4
0dense_420_matmul_readvariableop_dense_420_kernel3
/dense_420_biasadd_readvariableop_dense_420_bias
identity?? dense_415/BiasAdd/ReadVariableOp?"dense_415/Tensordot/ReadVariableOp? dense_416/BiasAdd/ReadVariableOp?"dense_416/Tensordot/ReadVariableOp? dense_417/BiasAdd/ReadVariableOp?"dense_417/Tensordot/ReadVariableOp? dense_418/BiasAdd/ReadVariableOp?"dense_418/Tensordot/ReadVariableOp? dense_419/BiasAdd/ReadVariableOp?"dense_419/Tensordot/ReadVariableOp? dense_420/BiasAdd/ReadVariableOp?dense_420/MatMul/ReadVariableOps
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulinputsdropout/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/dropout/Muld
dropout/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/dropout/Mul_1?
"dense_415/Tensordot/ReadVariableOpReadVariableOp3dense_415_tensordot_readvariableop_dense_415_kernel* 
_output_shapes
:
??*
dtype02$
"dense_415/Tensordot/ReadVariableOp~
dense_415/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_415/Tensordot/axes?
dense_415/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_415/Tensordot/free
dense_415/Tensordot/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_415/Tensordot/Shape?
!dense_415/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_415/Tensordot/GatherV2/axis?
dense_415/Tensordot/GatherV2GatherV2"dense_415/Tensordot/Shape:output:0!dense_415/Tensordot/free:output:0*dense_415/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_415/Tensordot/GatherV2?
#dense_415/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_415/Tensordot/GatherV2_1/axis?
dense_415/Tensordot/GatherV2_1GatherV2"dense_415/Tensordot/Shape:output:0!dense_415/Tensordot/axes:output:0,dense_415/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_415/Tensordot/GatherV2_1?
dense_415/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_415/Tensordot/Const?
dense_415/Tensordot/ProdProd%dense_415/Tensordot/GatherV2:output:0"dense_415/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_415/Tensordot/Prod?
dense_415/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_415/Tensordot/Const_1?
dense_415/Tensordot/Prod_1Prod'dense_415/Tensordot/GatherV2_1:output:0$dense_415/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_415/Tensordot/Prod_1?
dense_415/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_415/Tensordot/concat/axis?
dense_415/Tensordot/concatConcatV2!dense_415/Tensordot/free:output:0!dense_415/Tensordot/axes:output:0(dense_415/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_415/Tensordot/concat?
dense_415/Tensordot/stackPack!dense_415/Tensordot/Prod:output:0#dense_415/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_415/Tensordot/stack?
dense_415/Tensordot/transpose	Transposedropout/dropout/Mul_1:z:0#dense_415/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_415/Tensordot/transpose?
dense_415/Tensordot/ReshapeReshape!dense_415/Tensordot/transpose:y:0"dense_415/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_415/Tensordot/Reshape?
dense_415/Tensordot/MatMulMatMul$dense_415/Tensordot/Reshape:output:0*dense_415/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_415/Tensordot/MatMul?
dense_415/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_415/Tensordot/Const_2?
!dense_415/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_415/Tensordot/concat_1/axis?
dense_415/Tensordot/concat_1ConcatV2%dense_415/Tensordot/GatherV2:output:0$dense_415/Tensordot/Const_2:output:0*dense_415/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_415/Tensordot/concat_1?
dense_415/TensordotReshape$dense_415/Tensordot/MatMul:product:0%dense_415/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_415/Tensordot?
 dense_415/BiasAdd/ReadVariableOpReadVariableOp/dense_415_biasadd_readvariableop_dense_415_bias*
_output_shapes	
:?*
dtype02"
 dense_415/BiasAdd/ReadVariableOp?
dense_415/BiasAddBiasAdddense_415/Tensordot:output:0(dense_415/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_415/BiasAdd{
dense_415/TanhTanhdense_415/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_415/Tanh?
"dense_416/Tensordot/ReadVariableOpReadVariableOp3dense_416_tensordot_readvariableop_dense_416_kernel* 
_output_shapes
:
??*
dtype02$
"dense_416/Tensordot/ReadVariableOp~
dense_416/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_416/Tensordot/axes?
dense_416/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_416/Tensordot/freex
dense_416/Tensordot/ShapeShapedense_415/Tanh:y:0*
T0*
_output_shapes
:2
dense_416/Tensordot/Shape?
!dense_416/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_416/Tensordot/GatherV2/axis?
dense_416/Tensordot/GatherV2GatherV2"dense_416/Tensordot/Shape:output:0!dense_416/Tensordot/free:output:0*dense_416/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_416/Tensordot/GatherV2?
#dense_416/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_416/Tensordot/GatherV2_1/axis?
dense_416/Tensordot/GatherV2_1GatherV2"dense_416/Tensordot/Shape:output:0!dense_416/Tensordot/axes:output:0,dense_416/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_416/Tensordot/GatherV2_1?
dense_416/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_416/Tensordot/Const?
dense_416/Tensordot/ProdProd%dense_416/Tensordot/GatherV2:output:0"dense_416/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_416/Tensordot/Prod?
dense_416/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_416/Tensordot/Const_1?
dense_416/Tensordot/Prod_1Prod'dense_416/Tensordot/GatherV2_1:output:0$dense_416/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_416/Tensordot/Prod_1?
dense_416/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_416/Tensordot/concat/axis?
dense_416/Tensordot/concatConcatV2!dense_416/Tensordot/free:output:0!dense_416/Tensordot/axes:output:0(dense_416/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_416/Tensordot/concat?
dense_416/Tensordot/stackPack!dense_416/Tensordot/Prod:output:0#dense_416/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_416/Tensordot/stack?
dense_416/Tensordot/transpose	Transposedense_415/Tanh:y:0#dense_416/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_416/Tensordot/transpose?
dense_416/Tensordot/ReshapeReshape!dense_416/Tensordot/transpose:y:0"dense_416/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_416/Tensordot/Reshape?
dense_416/Tensordot/MatMulMatMul$dense_416/Tensordot/Reshape:output:0*dense_416/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_416/Tensordot/MatMul?
dense_416/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_416/Tensordot/Const_2?
!dense_416/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_416/Tensordot/concat_1/axis?
dense_416/Tensordot/concat_1ConcatV2%dense_416/Tensordot/GatherV2:output:0$dense_416/Tensordot/Const_2:output:0*dense_416/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_416/Tensordot/concat_1?
dense_416/TensordotReshape$dense_416/Tensordot/MatMul:product:0%dense_416/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_416/Tensordot?
 dense_416/BiasAdd/ReadVariableOpReadVariableOp/dense_416_biasadd_readvariableop_dense_416_bias*
_output_shapes	
:?*
dtype02"
 dense_416/BiasAdd/ReadVariableOp?
dense_416/BiasAddBiasAdddense_416/Tensordot:output:0(dense_416/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_416/BiasAdd{
dense_416/TanhTanhdense_416/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_416/Tanh?
"dense_417/Tensordot/ReadVariableOpReadVariableOp3dense_417_tensordot_readvariableop_dense_417_kernel* 
_output_shapes
:
??*
dtype02$
"dense_417/Tensordot/ReadVariableOp~
dense_417/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_417/Tensordot/axes?
dense_417/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_417/Tensordot/freex
dense_417/Tensordot/ShapeShapedense_416/Tanh:y:0*
T0*
_output_shapes
:2
dense_417/Tensordot/Shape?
!dense_417/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_417/Tensordot/GatherV2/axis?
dense_417/Tensordot/GatherV2GatherV2"dense_417/Tensordot/Shape:output:0!dense_417/Tensordot/free:output:0*dense_417/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_417/Tensordot/GatherV2?
#dense_417/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_417/Tensordot/GatherV2_1/axis?
dense_417/Tensordot/GatherV2_1GatherV2"dense_417/Tensordot/Shape:output:0!dense_417/Tensordot/axes:output:0,dense_417/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_417/Tensordot/GatherV2_1?
dense_417/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_417/Tensordot/Const?
dense_417/Tensordot/ProdProd%dense_417/Tensordot/GatherV2:output:0"dense_417/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_417/Tensordot/Prod?
dense_417/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_417/Tensordot/Const_1?
dense_417/Tensordot/Prod_1Prod'dense_417/Tensordot/GatherV2_1:output:0$dense_417/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_417/Tensordot/Prod_1?
dense_417/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_417/Tensordot/concat/axis?
dense_417/Tensordot/concatConcatV2!dense_417/Tensordot/free:output:0!dense_417/Tensordot/axes:output:0(dense_417/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_417/Tensordot/concat?
dense_417/Tensordot/stackPack!dense_417/Tensordot/Prod:output:0#dense_417/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_417/Tensordot/stack?
dense_417/Tensordot/transpose	Transposedense_416/Tanh:y:0#dense_417/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_417/Tensordot/transpose?
dense_417/Tensordot/ReshapeReshape!dense_417/Tensordot/transpose:y:0"dense_417/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_417/Tensordot/Reshape?
dense_417/Tensordot/MatMulMatMul$dense_417/Tensordot/Reshape:output:0*dense_417/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_417/Tensordot/MatMul?
dense_417/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_417/Tensordot/Const_2?
!dense_417/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_417/Tensordot/concat_1/axis?
dense_417/Tensordot/concat_1ConcatV2%dense_417/Tensordot/GatherV2:output:0$dense_417/Tensordot/Const_2:output:0*dense_417/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_417/Tensordot/concat_1?
dense_417/TensordotReshape$dense_417/Tensordot/MatMul:product:0%dense_417/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_417/Tensordot?
 dense_417/BiasAdd/ReadVariableOpReadVariableOp/dense_417_biasadd_readvariableop_dense_417_bias*
_output_shapes	
:?*
dtype02"
 dense_417/BiasAdd/ReadVariableOp?
dense_417/BiasAddBiasAdddense_417/Tensordot:output:0(dense_417/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_417/BiasAdd{
dense_417/TanhTanhdense_417/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_417/Tanh?
"dense_418/Tensordot/ReadVariableOpReadVariableOp3dense_418_tensordot_readvariableop_dense_418_kernel* 
_output_shapes
:
??*
dtype02$
"dense_418/Tensordot/ReadVariableOp~
dense_418/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_418/Tensordot/axes?
dense_418/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_418/Tensordot/freex
dense_418/Tensordot/ShapeShapedense_417/Tanh:y:0*
T0*
_output_shapes
:2
dense_418/Tensordot/Shape?
!dense_418/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_418/Tensordot/GatherV2/axis?
dense_418/Tensordot/GatherV2GatherV2"dense_418/Tensordot/Shape:output:0!dense_418/Tensordot/free:output:0*dense_418/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_418/Tensordot/GatherV2?
#dense_418/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_418/Tensordot/GatherV2_1/axis?
dense_418/Tensordot/GatherV2_1GatherV2"dense_418/Tensordot/Shape:output:0!dense_418/Tensordot/axes:output:0,dense_418/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_418/Tensordot/GatherV2_1?
dense_418/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_418/Tensordot/Const?
dense_418/Tensordot/ProdProd%dense_418/Tensordot/GatherV2:output:0"dense_418/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_418/Tensordot/Prod?
dense_418/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_418/Tensordot/Const_1?
dense_418/Tensordot/Prod_1Prod'dense_418/Tensordot/GatherV2_1:output:0$dense_418/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_418/Tensordot/Prod_1?
dense_418/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_418/Tensordot/concat/axis?
dense_418/Tensordot/concatConcatV2!dense_418/Tensordot/free:output:0!dense_418/Tensordot/axes:output:0(dense_418/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_418/Tensordot/concat?
dense_418/Tensordot/stackPack!dense_418/Tensordot/Prod:output:0#dense_418/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_418/Tensordot/stack?
dense_418/Tensordot/transpose	Transposedense_417/Tanh:y:0#dense_418/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_418/Tensordot/transpose?
dense_418/Tensordot/ReshapeReshape!dense_418/Tensordot/transpose:y:0"dense_418/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_418/Tensordot/Reshape?
dense_418/Tensordot/MatMulMatMul$dense_418/Tensordot/Reshape:output:0*dense_418/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_418/Tensordot/MatMul?
dense_418/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_418/Tensordot/Const_2?
!dense_418/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_418/Tensordot/concat_1/axis?
dense_418/Tensordot/concat_1ConcatV2%dense_418/Tensordot/GatherV2:output:0$dense_418/Tensordot/Const_2:output:0*dense_418/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_418/Tensordot/concat_1?
dense_418/TensordotReshape$dense_418/Tensordot/MatMul:product:0%dense_418/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_418/Tensordot?
 dense_418/BiasAdd/ReadVariableOpReadVariableOp/dense_418_biasadd_readvariableop_dense_418_bias*
_output_shapes	
:?*
dtype02"
 dense_418/BiasAdd/ReadVariableOp?
dense_418/BiasAddBiasAdddense_418/Tensordot:output:0(dense_418/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_418/BiasAdd{
dense_418/TanhTanhdense_418/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_418/Tanh?
"dense_419/Tensordot/ReadVariableOpReadVariableOp3dense_419_tensordot_readvariableop_dense_419_kernel*
_output_shapes
:	?*
dtype02$
"dense_419/Tensordot/ReadVariableOp~
dense_419/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_419/Tensordot/axes?
dense_419/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_419/Tensordot/freex
dense_419/Tensordot/ShapeShapedense_418/Tanh:y:0*
T0*
_output_shapes
:2
dense_419/Tensordot/Shape?
!dense_419/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_419/Tensordot/GatherV2/axis?
dense_419/Tensordot/GatherV2GatherV2"dense_419/Tensordot/Shape:output:0!dense_419/Tensordot/free:output:0*dense_419/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_419/Tensordot/GatherV2?
#dense_419/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_419/Tensordot/GatherV2_1/axis?
dense_419/Tensordot/GatherV2_1GatherV2"dense_419/Tensordot/Shape:output:0!dense_419/Tensordot/axes:output:0,dense_419/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_419/Tensordot/GatherV2_1?
dense_419/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_419/Tensordot/Const?
dense_419/Tensordot/ProdProd%dense_419/Tensordot/GatherV2:output:0"dense_419/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_419/Tensordot/Prod?
dense_419/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_419/Tensordot/Const_1?
dense_419/Tensordot/Prod_1Prod'dense_419/Tensordot/GatherV2_1:output:0$dense_419/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_419/Tensordot/Prod_1?
dense_419/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_419/Tensordot/concat/axis?
dense_419/Tensordot/concatConcatV2!dense_419/Tensordot/free:output:0!dense_419/Tensordot/axes:output:0(dense_419/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_419/Tensordot/concat?
dense_419/Tensordot/stackPack!dense_419/Tensordot/Prod:output:0#dense_419/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_419/Tensordot/stack?
dense_419/Tensordot/transpose	Transposedense_418/Tanh:y:0#dense_419/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_419/Tensordot/transpose?
dense_419/Tensordot/ReshapeReshape!dense_419/Tensordot/transpose:y:0"dense_419/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_419/Tensordot/Reshape?
dense_419/Tensordot/MatMulMatMul$dense_419/Tensordot/Reshape:output:0*dense_419/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_419/Tensordot/MatMul?
dense_419/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_419/Tensordot/Const_2?
!dense_419/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_419/Tensordot/concat_1/axis?
dense_419/Tensordot/concat_1ConcatV2%dense_419/Tensordot/GatherV2:output:0$dense_419/Tensordot/Const_2:output:0*dense_419/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_419/Tensordot/concat_1?
dense_419/TensordotReshape$dense_419/Tensordot/MatMul:product:0%dense_419/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_419/Tensordot?
 dense_419/BiasAdd/ReadVariableOpReadVariableOp/dense_419_biasadd_readvariableop_dense_419_bias*
_output_shapes
:*
dtype02"
 dense_419/BiasAdd/ReadVariableOp?
dense_419/BiasAddBiasAdddense_419/Tensordot:output:0(dense_419/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_419/BiasAddz
dense_419/TanhTanhdense_419/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense_419/Tanhu
flatten_51/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_51/Const?
flatten_51/ReshapeReshapedense_419/Tanh:y:0flatten_51/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_51/Reshape?
dense_420/MatMul/ReadVariableOpReadVariableOp0dense_420_matmul_readvariableop_dense_420_kernel*
_output_shapes

:*
dtype02!
dense_420/MatMul/ReadVariableOp?
dense_420/MatMulMatMulflatten_51/Reshape:output:0'dense_420/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_420/MatMul?
 dense_420/BiasAdd/ReadVariableOpReadVariableOp/dense_420_biasadd_readvariableop_dense_420_bias*
_output_shapes
:*
dtype02"
 dense_420/BiasAdd/ReadVariableOp?
dense_420/BiasAddBiasAdddense_420/MatMul:product:0(dense_420/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_420/BiasAdd
dense_420/SigmoidSigmoiddense_420/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_420/Sigmoid?
IdentityIdentitydense_420/Sigmoid:y:0!^dense_415/BiasAdd/ReadVariableOp#^dense_415/Tensordot/ReadVariableOp!^dense_416/BiasAdd/ReadVariableOp#^dense_416/Tensordot/ReadVariableOp!^dense_417/BiasAdd/ReadVariableOp#^dense_417/Tensordot/ReadVariableOp!^dense_418/BiasAdd/ReadVariableOp#^dense_418/Tensordot/ReadVariableOp!^dense_419/BiasAdd/ReadVariableOp#^dense_419/Tensordot/ReadVariableOp!^dense_420/BiasAdd/ReadVariableOp ^dense_420/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::2D
 dense_415/BiasAdd/ReadVariableOp dense_415/BiasAdd/ReadVariableOp2H
"dense_415/Tensordot/ReadVariableOp"dense_415/Tensordot/ReadVariableOp2D
 dense_416/BiasAdd/ReadVariableOp dense_416/BiasAdd/ReadVariableOp2H
"dense_416/Tensordot/ReadVariableOp"dense_416/Tensordot/ReadVariableOp2D
 dense_417/BiasAdd/ReadVariableOp dense_417/BiasAdd/ReadVariableOp2H
"dense_417/Tensordot/ReadVariableOp"dense_417/Tensordot/ReadVariableOp2D
 dense_418/BiasAdd/ReadVariableOp dense_418/BiasAdd/ReadVariableOp2H
"dense_418/Tensordot/ReadVariableOp"dense_418/Tensordot/ReadVariableOp2D
 dense_419/BiasAdd/ReadVariableOp dense_419/BiasAdd/ReadVariableOp2H
"dense_419/Tensordot/ReadVariableOp"dense_419/Tensordot/ReadVariableOp2D
 dense_420/BiasAdd/ReadVariableOp dense_420/BiasAdd/ReadVariableOp2B
dense_420/MatMul/ReadVariableOpdense_420/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?p
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60989933

inputs7
3dense_412_tensordot_readvariableop_dense_412_kernel3
/dense_412_biasadd_readvariableop_dense_412_bias7
3dense_413_tensordot_readvariableop_dense_413_kernel3
/dense_413_biasadd_readvariableop_dense_413_bias7
3dense_414_tensordot_readvariableop_dense_414_kernel3
/dense_414_biasadd_readvariableop_dense_414_bias
identity?? dense_412/BiasAdd/ReadVariableOp?"dense_412/Tensordot/ReadVariableOp? dense_413/BiasAdd/ReadVariableOp?"dense_413/Tensordot/ReadVariableOp? dense_414/BiasAdd/ReadVariableOp?"dense_414/Tensordot/ReadVariableOp?
"dense_412/Tensordot/ReadVariableOpReadVariableOp3dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype02$
"dense_412/Tensordot/ReadVariableOp~
dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_412/Tensordot/axes?
dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_412/Tensordot/freel
dense_412/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_412/Tensordot/Shape?
!dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/GatherV2/axis?
dense_412/Tensordot/GatherV2GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/free:output:0*dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_412/Tensordot/GatherV2?
#dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_412/Tensordot/GatherV2_1/axis?
dense_412/Tensordot/GatherV2_1GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/axes:output:0,dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_412/Tensordot/GatherV2_1?
dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const?
dense_412/Tensordot/ProdProd%dense_412/Tensordot/GatherV2:output:0"dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod?
dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_1?
dense_412/Tensordot/Prod_1Prod'dense_412/Tensordot/GatherV2_1:output:0$dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod_1?
dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_412/Tensordot/concat/axis?
dense_412/Tensordot/concatConcatV2!dense_412/Tensordot/free:output:0!dense_412/Tensordot/axes:output:0(dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat?
dense_412/Tensordot/stackPack!dense_412/Tensordot/Prod:output:0#dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/stack?
dense_412/Tensordot/transpose	Transposeinputs#dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_412/Tensordot/transpose?
dense_412/Tensordot/ReshapeReshape!dense_412/Tensordot/transpose:y:0"dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_412/Tensordot/Reshape?
dense_412/Tensordot/MatMulMatMul$dense_412/Tensordot/Reshape:output:0*dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_412/Tensordot/MatMul?
dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_2?
!dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/concat_1/axis?
dense_412/Tensordot/concat_1ConcatV2%dense_412/Tensordot/GatherV2:output:0$dense_412/Tensordot/Const_2:output:0*dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat_1?
dense_412/TensordotReshape$dense_412/Tensordot/MatMul:product:0%dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tensordot?
 dense_412/BiasAdd/ReadVariableOpReadVariableOp/dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02"
 dense_412/BiasAdd/ReadVariableOp?
dense_412/BiasAddBiasAdddense_412/Tensordot:output:0(dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_412/BiasAddz
dense_412/TanhTanhdense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tanh?
"dense_413/Tensordot/ReadVariableOpReadVariableOp3dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_413/Tensordot/ReadVariableOp~
dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_413/Tensordot/axes?
dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_413/Tensordot/freex
dense_413/Tensordot/ShapeShapedense_412/Tanh:y:0*
T0*
_output_shapes
:2
dense_413/Tensordot/Shape?
!dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/GatherV2/axis?
dense_413/Tensordot/GatherV2GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/free:output:0*dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_413/Tensordot/GatherV2?
#dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_413/Tensordot/GatherV2_1/axis?
dense_413/Tensordot/GatherV2_1GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/axes:output:0,dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_413/Tensordot/GatherV2_1?
dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const?
dense_413/Tensordot/ProdProd%dense_413/Tensordot/GatherV2:output:0"dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod?
dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const_1?
dense_413/Tensordot/Prod_1Prod'dense_413/Tensordot/GatherV2_1:output:0$dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod_1?
dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_413/Tensordot/concat/axis?
dense_413/Tensordot/concatConcatV2!dense_413/Tensordot/free:output:0!dense_413/Tensordot/axes:output:0(dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat?
dense_413/Tensordot/stackPack!dense_413/Tensordot/Prod:output:0#dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/stack?
dense_413/Tensordot/transpose	Transposedense_412/Tanh:y:0#dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_413/Tensordot/transpose?
dense_413/Tensordot/ReshapeReshape!dense_413/Tensordot/transpose:y:0"dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_413/Tensordot/Reshape?
dense_413/Tensordot/MatMulMatMul$dense_413/Tensordot/Reshape:output:0*dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_413/Tensordot/MatMul?
dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_413/Tensordot/Const_2?
!dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/concat_1/axis?
dense_413/Tensordot/concat_1ConcatV2%dense_413/Tensordot/GatherV2:output:0$dense_413/Tensordot/Const_2:output:0*dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat_1?
dense_413/TensordotReshape$dense_413/Tensordot/MatMul:product:0%dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tensordot?
 dense_413/BiasAdd/ReadVariableOpReadVariableOp/dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02"
 dense_413/BiasAdd/ReadVariableOp?
dense_413/BiasAddBiasAdddense_413/Tensordot:output:0(dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_413/BiasAdd{
dense_413/TanhTanhdense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tanh?
"dense_414/Tensordot/ReadVariableOpReadVariableOp3dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype02$
"dense_414/Tensordot/ReadVariableOp~
dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_414/Tensordot/axes?
dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_414/Tensordot/freex
dense_414/Tensordot/ShapeShapedense_413/Tanh:y:0*
T0*
_output_shapes
:2
dense_414/Tensordot/Shape?
!dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/GatherV2/axis?
dense_414/Tensordot/GatherV2GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/free:output:0*dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_414/Tensordot/GatherV2?
#dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_414/Tensordot/GatherV2_1/axis?
dense_414/Tensordot/GatherV2_1GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/axes:output:0,dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_414/Tensordot/GatherV2_1?
dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const?
dense_414/Tensordot/ProdProd%dense_414/Tensordot/GatherV2:output:0"dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod?
dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const_1?
dense_414/Tensordot/Prod_1Prod'dense_414/Tensordot/GatherV2_1:output:0$dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod_1?
dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_414/Tensordot/concat/axis?
dense_414/Tensordot/concatConcatV2!dense_414/Tensordot/free:output:0!dense_414/Tensordot/axes:output:0(dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat?
dense_414/Tensordot/stackPack!dense_414/Tensordot/Prod:output:0#dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/stack?
dense_414/Tensordot/transpose	Transposedense_413/Tanh:y:0#dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot/transpose?
dense_414/Tensordot/ReshapeReshape!dense_414/Tensordot/transpose:y:0"dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_414/Tensordot/Reshape?
dense_414/Tensordot/MatMulMatMul$dense_414/Tensordot/Reshape:output:0*dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_414/Tensordot/MatMul?
dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_414/Tensordot/Const_2?
!dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/concat_1/axis?
dense_414/Tensordot/concat_1ConcatV2%dense_414/Tensordot/GatherV2:output:0$dense_414/Tensordot/Const_2:output:0*dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat_1?
dense_414/TensordotReshape$dense_414/Tensordot/MatMul:product:0%dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot?
 dense_414/BiasAdd/ReadVariableOpReadVariableOp/dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02"
 dense_414/BiasAdd/ReadVariableOp?
dense_414/BiasAddBiasAdddense_414/Tensordot:output:0(dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_414/BiasAdd?
IdentityIdentitydense_414/BiasAdd:output:0!^dense_412/BiasAdd/ReadVariableOp#^dense_412/Tensordot/ReadVariableOp!^dense_413/BiasAdd/ReadVariableOp#^dense_413/Tensordot/ReadVariableOp!^dense_414/BiasAdd/ReadVariableOp#^dense_414/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_412/BiasAdd/ReadVariableOp dense_412/BiasAdd/ReadVariableOp2H
"dense_412/Tensordot/ReadVariableOp"dense_412/Tensordot/ReadVariableOp2D
 dense_413/BiasAdd/ReadVariableOp dense_413/BiasAdd/ReadVariableOp2H
"dense_413/Tensordot/ReadVariableOp"dense_413/Tensordot/ReadVariableOp2D
 dense_414/BiasAdd/ReadVariableOp dense_414/BiasAdd/ReadVariableOp2H
"dense_414/Tensordot/ReadVariableOp"dense_414/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?p
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60989849

inputs7
3dense_412_tensordot_readvariableop_dense_412_kernel3
/dense_412_biasadd_readvariableop_dense_412_bias7
3dense_413_tensordot_readvariableop_dense_413_kernel3
/dense_413_biasadd_readvariableop_dense_413_bias7
3dense_414_tensordot_readvariableop_dense_414_kernel3
/dense_414_biasadd_readvariableop_dense_414_bias
identity?? dense_412/BiasAdd/ReadVariableOp?"dense_412/Tensordot/ReadVariableOp? dense_413/BiasAdd/ReadVariableOp?"dense_413/Tensordot/ReadVariableOp? dense_414/BiasAdd/ReadVariableOp?"dense_414/Tensordot/ReadVariableOp?
"dense_412/Tensordot/ReadVariableOpReadVariableOp3dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype02$
"dense_412/Tensordot/ReadVariableOp~
dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_412/Tensordot/axes?
dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_412/Tensordot/freel
dense_412/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_412/Tensordot/Shape?
!dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/GatherV2/axis?
dense_412/Tensordot/GatherV2GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/free:output:0*dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_412/Tensordot/GatherV2?
#dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_412/Tensordot/GatherV2_1/axis?
dense_412/Tensordot/GatherV2_1GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/axes:output:0,dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_412/Tensordot/GatherV2_1?
dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const?
dense_412/Tensordot/ProdProd%dense_412/Tensordot/GatherV2:output:0"dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod?
dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_1?
dense_412/Tensordot/Prod_1Prod'dense_412/Tensordot/GatherV2_1:output:0$dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod_1?
dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_412/Tensordot/concat/axis?
dense_412/Tensordot/concatConcatV2!dense_412/Tensordot/free:output:0!dense_412/Tensordot/axes:output:0(dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat?
dense_412/Tensordot/stackPack!dense_412/Tensordot/Prod:output:0#dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/stack?
dense_412/Tensordot/transpose	Transposeinputs#dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_412/Tensordot/transpose?
dense_412/Tensordot/ReshapeReshape!dense_412/Tensordot/transpose:y:0"dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_412/Tensordot/Reshape?
dense_412/Tensordot/MatMulMatMul$dense_412/Tensordot/Reshape:output:0*dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_412/Tensordot/MatMul?
dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_2?
!dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/concat_1/axis?
dense_412/Tensordot/concat_1ConcatV2%dense_412/Tensordot/GatherV2:output:0$dense_412/Tensordot/Const_2:output:0*dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat_1?
dense_412/TensordotReshape$dense_412/Tensordot/MatMul:product:0%dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tensordot?
 dense_412/BiasAdd/ReadVariableOpReadVariableOp/dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02"
 dense_412/BiasAdd/ReadVariableOp?
dense_412/BiasAddBiasAdddense_412/Tensordot:output:0(dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_412/BiasAddz
dense_412/TanhTanhdense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tanh?
"dense_413/Tensordot/ReadVariableOpReadVariableOp3dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_413/Tensordot/ReadVariableOp~
dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_413/Tensordot/axes?
dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_413/Tensordot/freex
dense_413/Tensordot/ShapeShapedense_412/Tanh:y:0*
T0*
_output_shapes
:2
dense_413/Tensordot/Shape?
!dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/GatherV2/axis?
dense_413/Tensordot/GatherV2GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/free:output:0*dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_413/Tensordot/GatherV2?
#dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_413/Tensordot/GatherV2_1/axis?
dense_413/Tensordot/GatherV2_1GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/axes:output:0,dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_413/Tensordot/GatherV2_1?
dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const?
dense_413/Tensordot/ProdProd%dense_413/Tensordot/GatherV2:output:0"dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod?
dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const_1?
dense_413/Tensordot/Prod_1Prod'dense_413/Tensordot/GatherV2_1:output:0$dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod_1?
dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_413/Tensordot/concat/axis?
dense_413/Tensordot/concatConcatV2!dense_413/Tensordot/free:output:0!dense_413/Tensordot/axes:output:0(dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat?
dense_413/Tensordot/stackPack!dense_413/Tensordot/Prod:output:0#dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/stack?
dense_413/Tensordot/transpose	Transposedense_412/Tanh:y:0#dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_413/Tensordot/transpose?
dense_413/Tensordot/ReshapeReshape!dense_413/Tensordot/transpose:y:0"dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_413/Tensordot/Reshape?
dense_413/Tensordot/MatMulMatMul$dense_413/Tensordot/Reshape:output:0*dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_413/Tensordot/MatMul?
dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_413/Tensordot/Const_2?
!dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/concat_1/axis?
dense_413/Tensordot/concat_1ConcatV2%dense_413/Tensordot/GatherV2:output:0$dense_413/Tensordot/Const_2:output:0*dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat_1?
dense_413/TensordotReshape$dense_413/Tensordot/MatMul:product:0%dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tensordot?
 dense_413/BiasAdd/ReadVariableOpReadVariableOp/dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02"
 dense_413/BiasAdd/ReadVariableOp?
dense_413/BiasAddBiasAdddense_413/Tensordot:output:0(dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_413/BiasAdd{
dense_413/TanhTanhdense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tanh?
"dense_414/Tensordot/ReadVariableOpReadVariableOp3dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype02$
"dense_414/Tensordot/ReadVariableOp~
dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_414/Tensordot/axes?
dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_414/Tensordot/freex
dense_414/Tensordot/ShapeShapedense_413/Tanh:y:0*
T0*
_output_shapes
:2
dense_414/Tensordot/Shape?
!dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/GatherV2/axis?
dense_414/Tensordot/GatherV2GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/free:output:0*dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_414/Tensordot/GatherV2?
#dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_414/Tensordot/GatherV2_1/axis?
dense_414/Tensordot/GatherV2_1GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/axes:output:0,dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_414/Tensordot/GatherV2_1?
dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const?
dense_414/Tensordot/ProdProd%dense_414/Tensordot/GatherV2:output:0"dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod?
dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const_1?
dense_414/Tensordot/Prod_1Prod'dense_414/Tensordot/GatherV2_1:output:0$dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod_1?
dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_414/Tensordot/concat/axis?
dense_414/Tensordot/concatConcatV2!dense_414/Tensordot/free:output:0!dense_414/Tensordot/axes:output:0(dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat?
dense_414/Tensordot/stackPack!dense_414/Tensordot/Prod:output:0#dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/stack?
dense_414/Tensordot/transpose	Transposedense_413/Tanh:y:0#dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot/transpose?
dense_414/Tensordot/ReshapeReshape!dense_414/Tensordot/transpose:y:0"dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_414/Tensordot/Reshape?
dense_414/Tensordot/MatMulMatMul$dense_414/Tensordot/Reshape:output:0*dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_414/Tensordot/MatMul?
dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_414/Tensordot/Const_2?
!dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/concat_1/axis?
dense_414/Tensordot/concat_1ConcatV2%dense_414/Tensordot/GatherV2:output:0$dense_414/Tensordot/Const_2:output:0*dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat_1?
dense_414/TensordotReshape$dense_414/Tensordot/MatMul:product:0%dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot?
 dense_414/BiasAdd/ReadVariableOpReadVariableOp/dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02"
 dense_414/BiasAdd/ReadVariableOp?
dense_414/BiasAddBiasAdddense_414/Tensordot:output:0(dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_414/BiasAdd?
IdentityIdentitydense_414/BiasAdd:output:0!^dense_412/BiasAdd/ReadVariableOp#^dense_412/Tensordot/ReadVariableOp!^dense_413/BiasAdd/ReadVariableOp#^dense_413/Tensordot/ReadVariableOp!^dense_414/BiasAdd/ReadVariableOp#^dense_414/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_412/BiasAdd/ReadVariableOp dense_412/BiasAdd/ReadVariableOp2H
"dense_412/Tensordot/ReadVariableOp"dense_412/Tensordot/ReadVariableOp2D
 dense_413/BiasAdd/ReadVariableOp dense_413/BiasAdd/ReadVariableOp2H
"dense_413/Tensordot/ReadVariableOp"dense_413/Tensordot/ReadVariableOp2D
 dense_414/BiasAdd/ReadVariableOp dense_414/BiasAdd/ReadVariableOp2H
"dense_414/Tensordot/ReadVariableOp"dense_414/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_layer_call_fn_60990619

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_609884832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_model_960_layer_call_fn_60989765

inputs
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_biasdense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_960_layer_call_and_return_conditional_losses_609892082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_412_layer_call_and_return_conditional_losses_60988307

inputs-
)tensordot_readvariableop_dense_412_kernel)
%biasadd_readvariableop_dense_412_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_412_kernel*
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
:??????????2
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
:????????? 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_417_layer_call_fn_60990738

inputs
dense_417_kernel
dense_417_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_417_kerneldense_417_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_417_layer_call_and_return_conditional_losses_609886182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988418
input_11
dense_412_dense_412_kernel
dense_412_dense_412_bias
dense_413_dense_413_kernel
dense_413_dense_413_bias
dense_414_dense_414_kernel
dense_414_dense_414_bias
identity??!dense_412/StatefulPartitionedCall?!dense_413/StatefulPartitionedCall?!dense_414/StatefulPartitionedCall?
!dense_412/StatefulPartitionedCallStatefulPartitionedCallinput_11dense_412_dense_412_kerneldense_412_dense_412_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_412_layer_call_and_return_conditional_losses_609883072#
!dense_412/StatefulPartitionedCall?
!dense_413/StatefulPartitionedCallStatefulPartitionedCall*dense_412/StatefulPartitionedCall:output:0dense_413_dense_413_kerneldense_413_dense_413_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_413_layer_call_and_return_conditional_losses_609883502#
!dense_413/StatefulPartitionedCall?
!dense_414/StatefulPartitionedCallStatefulPartitionedCall*dense_413/StatefulPartitionedCall:output:0dense_414_dense_414_kerneldense_414_dense_414_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_414_layer_call_and_return_conditional_losses_609883922#
!dense_414/StatefulPartitionedCall?
IdentityIdentity*dense_414/StatefulPartitionedCall:output:0"^dense_412/StatefulPartitionedCall"^dense_413/StatefulPartitionedCall"^dense_414/StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2F
!dense_412/StatefulPartitionedCall!dense_412/StatefulPartitionedCall2F
!dense_413/StatefulPartitionedCall!dense_413/StatefulPartitionedCall2F
!dense_414/StatefulPartitionedCall!dense_414/StatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
? 
?
G__inference_dense_412_layer_call_and_return_conditional_losses_60990515

inputs-
)tensordot_readvariableop_dense_412_kernel)
%biasadd_readvariableop_dense_412_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_412_kernel*
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
:??????????2
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
:????????? 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?(
$__inference__traced_restore_60991296
file_prefix+
'assignvariableop_training_318_adam_iter/
+assignvariableop_1_training_318_adam_beta_1/
+assignvariableop_2_training_318_adam_beta_2.
*assignvariableop_3_training_318_adam_decay6
2assignvariableop_4_training_318_adam_learning_rate'
#assignvariableop_5_dense_412_kernel%
!assignvariableop_6_dense_412_bias'
#assignvariableop_7_dense_413_kernel%
!assignvariableop_8_dense_413_bias'
#assignvariableop_9_dense_414_kernel&
"assignvariableop_10_dense_414_bias(
$assignvariableop_11_dense_415_kernel&
"assignvariableop_12_dense_415_bias(
$assignvariableop_13_dense_416_kernel&
"assignvariableop_14_dense_416_bias(
$assignvariableop_15_dense_417_kernel&
"assignvariableop_16_dense_417_bias(
$assignvariableop_17_dense_418_kernel&
"assignvariableop_18_dense_418_bias(
$assignvariableop_19_dense_419_kernel&
"assignvariableop_20_dense_419_bias(
$assignvariableop_21_dense_420_kernel&
"assignvariableop_22_dense_420_bias.
*assignvariableop_23_training_130_adam_iter0
,assignvariableop_24_training_130_adam_beta_10
,assignvariableop_25_training_130_adam_beta_2/
+assignvariableop_26_training_130_adam_decay7
3assignvariableop_27_training_130_adam_learning_rate!
assignvariableop_28_total_329!
assignvariableop_29_count_329!
assignvariableop_30_total_139!
assignvariableop_31_count_139!
assignvariableop_32_total_140!
assignvariableop_33_count_140<
8assignvariableop_34_training_318_adam_dense_412_kernel_m:
6assignvariableop_35_training_318_adam_dense_412_bias_m<
8assignvariableop_36_training_318_adam_dense_413_kernel_m:
6assignvariableop_37_training_318_adam_dense_413_bias_m<
8assignvariableop_38_training_318_adam_dense_414_kernel_m:
6assignvariableop_39_training_318_adam_dense_414_bias_m<
8assignvariableop_40_training_318_adam_dense_412_kernel_v:
6assignvariableop_41_training_318_adam_dense_412_bias_v<
8assignvariableop_42_training_318_adam_dense_413_kernel_v:
6assignvariableop_43_training_318_adam_dense_413_bias_v<
8assignvariableop_44_training_318_adam_dense_414_kernel_v:
6assignvariableop_45_training_318_adam_dense_414_bias_v<
8assignvariableop_46_training_130_adam_dense_415_kernel_m:
6assignvariableop_47_training_130_adam_dense_415_bias_m<
8assignvariableop_48_training_130_adam_dense_416_kernel_m:
6assignvariableop_49_training_130_adam_dense_416_bias_m<
8assignvariableop_50_training_130_adam_dense_417_kernel_m:
6assignvariableop_51_training_130_adam_dense_417_bias_m<
8assignvariableop_52_training_130_adam_dense_418_kernel_m:
6assignvariableop_53_training_130_adam_dense_418_bias_m<
8assignvariableop_54_training_130_adam_dense_419_kernel_m:
6assignvariableop_55_training_130_adam_dense_419_bias_m<
8assignvariableop_56_training_130_adam_dense_420_kernel_m:
6assignvariableop_57_training_130_adam_dense_420_bias_m<
8assignvariableop_58_training_130_adam_dense_415_kernel_v:
6assignvariableop_59_training_130_adam_dense_415_bias_v<
8assignvariableop_60_training_130_adam_dense_416_kernel_v:
6assignvariableop_61_training_130_adam_dense_416_bias_v<
8assignvariableop_62_training_130_adam_dense_417_kernel_v:
6assignvariableop_63_training_130_adam_dense_417_bias_v<
8assignvariableop_64_training_130_adam_dense_418_kernel_v:
6assignvariableop_65_training_130_adam_dense_418_bias_v<
8assignvariableop_66_training_130_adam_dense_419_kernel_v:
6assignvariableop_67_training_130_adam_dense_419_bias_v<
8assignvariableop_68_training_130_adam_dense_420_kernel_v:
6assignvariableop_69_training_130_adam_dense_420_bias_v
identity_71??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*?&
value?&B?&GB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/16/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/6/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/7/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/8/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/9/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/10/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/13/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/14/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/15/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/16/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*?
value?B?GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*U
dtypesK
I2G		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_training_318_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_training_318_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp+assignvariableop_2_training_318_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_training_318_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_training_318_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_412_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_412_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_413_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_413_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_414_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_414_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_415_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_415_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dense_416_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_416_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_417_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_417_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_dense_418_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_418_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_dense_419_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_419_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_420_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_420_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_training_130_adam_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_training_130_adam_beta_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_training_130_adam_beta_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_training_130_adam_decayIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp3assignvariableop_27_training_130_adam_learning_rateIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_329Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_329Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_139Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_139Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_140Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_140Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp8assignvariableop_34_training_318_adam_dense_412_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_training_318_adam_dense_412_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp8assignvariableop_36_training_318_adam_dense_413_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp6assignvariableop_37_training_318_adam_dense_413_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp8assignvariableop_38_training_318_adam_dense_414_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_training_318_adam_dense_414_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp8assignvariableop_40_training_318_adam_dense_412_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp6assignvariableop_41_training_318_adam_dense_412_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp8assignvariableop_42_training_318_adam_dense_413_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_training_318_adam_dense_413_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp8assignvariableop_44_training_318_adam_dense_414_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp6assignvariableop_45_training_318_adam_dense_414_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp8assignvariableop_46_training_130_adam_dense_415_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp6assignvariableop_47_training_130_adam_dense_415_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp8assignvariableop_48_training_130_adam_dense_416_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp6assignvariableop_49_training_130_adam_dense_416_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp8assignvariableop_50_training_130_adam_dense_417_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_training_130_adam_dense_417_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp8assignvariableop_52_training_130_adam_dense_418_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp6assignvariableop_53_training_130_adam_dense_418_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp8assignvariableop_54_training_130_adam_dense_419_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp6assignvariableop_55_training_130_adam_dense_419_bias_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp8assignvariableop_56_training_130_adam_dense_420_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp6assignvariableop_57_training_130_adam_dense_420_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp8assignvariableop_58_training_130_adam_dense_415_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp6assignvariableop_59_training_130_adam_dense_415_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp8assignvariableop_60_training_130_adam_dense_416_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp6assignvariableop_61_training_130_adam_dense_416_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp8assignvariableop_62_training_130_adam_dense_417_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp6assignvariableop_63_training_130_adam_dense_417_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp8assignvariableop_64_training_130_adam_dense_418_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp6assignvariableop_65_training_130_adam_dense_418_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp8assignvariableop_66_training_130_adam_dense_419_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp6assignvariableop_67_training_130_adam_dense_419_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp8assignvariableop_68_training_130_adam_dense_420_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_training_130_adam_dense_420_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_699
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_70Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_70?
Identity_71IdentityIdentity_70:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_71"#
identity_71Identity_71:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
G__inference_dense_414_layer_call_and_return_conditional_losses_60990590

inputs-
)tensordot_readvariableop_dense_414_kernel)
%biasadd_readvariableop_dense_414_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_414_kernel* 
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_dense_420_layer_call_and_return_conditional_losses_60990836

inputs*
&matmul_readvariableop_dense_420_kernel)
%biasadd_readvariableop_dense_420_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp&matmul_readvariableop_dense_420_kernel*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_420_bias*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_model_960_layer_call_fn_60989229
input_11
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11dense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_biasdense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_960_layer_call_and_return_conditional_losses_609892082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
?
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988434

inputs
dense_412_dense_412_kernel
dense_412_dense_412_bias
dense_413_dense_413_kernel
dense_413_dense_413_bias
dense_414_dense_414_kernel
dense_414_dense_414_bias
identity??!dense_412/StatefulPartitionedCall?!dense_413/StatefulPartitionedCall?!dense_414/StatefulPartitionedCall?
!dense_412/StatefulPartitionedCallStatefulPartitionedCallinputsdense_412_dense_412_kerneldense_412_dense_412_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_412_layer_call_and_return_conditional_losses_609883072#
!dense_412/StatefulPartitionedCall?
!dense_413/StatefulPartitionedCallStatefulPartitionedCall*dense_412/StatefulPartitionedCall:output:0dense_413_dense_413_kerneldense_413_dense_413_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_413_layer_call_and_return_conditional_losses_609883502#
!dense_413/StatefulPartitionedCall?
!dense_414/StatefulPartitionedCallStatefulPartitionedCall*dense_413/StatefulPartitionedCall:output:0dense_414_dense_414_kerneldense_414_dense_414_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_414_layer_call_and_return_conditional_losses_609883922#
!dense_414/StatefulPartitionedCall?
IdentityIdentity*dense_414/StatefulPartitionedCall:output:0"^dense_412/StatefulPartitionedCall"^dense_413/StatefulPartitionedCall"^dense_414/StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2F
!dense_412/StatefulPartitionedCall!dense_412/StatefulPartitionedCall2F
!dense_413/StatefulPartitionedCall!dense_413/StatefulPartitionedCall2F
!dense_414/StatefulPartitionedCall!dense_414/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_417_layer_call_and_return_conditional_losses_60990731

inputs-
)tensordot_readvariableop_dense_417_kernel)
%biasadd_readvariableop_dense_417_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_417_kernel* 
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_417_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
0__inference_discriminator_layer_call_fn_60988820
input_12
dense_415_kernel
dense_415_bias
dense_416_kernel
dense_416_bias
dense_417_kernel
dense_417_bias
dense_418_kernel
dense_418_bias
dense_419_kernel
dense_419_bias
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12dense_415_kerneldense_415_biasdense_416_kerneldense_416_biasdense_417_kerneldense_417_biasdense_418_kerneldense_418_biasdense_419_kerneldense_419_biasdense_420_kerneldense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609888052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_12
?
?
.__inference_autoencoder_layer_call_fn_60989955

inputs
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609884582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_420_layer_call_fn_60990843

inputs
dense_420_kernel
dense_420_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_420_kerneldense_420_bias*
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
G__inference_dense_420_layer_call_and_return_conditional_losses_609887412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_960_layer_call_and_return_conditional_losses_60989161

inputs 
autoencoder_dense_412_kernel
autoencoder_dense_412_bias 
autoencoder_dense_413_kernel
autoencoder_dense_413_bias 
autoencoder_dense_414_kernel
autoencoder_dense_414_bias"
discriminator_dense_415_kernel 
discriminator_dense_415_bias"
discriminator_dense_416_kernel 
discriminator_dense_416_bias"
discriminator_dense_417_kernel 
discriminator_dense_417_bias"
discriminator_dense_418_kernel 
discriminator_dense_418_bias"
discriminator_dense_419_kernel 
discriminator_dense_419_bias"
discriminator_dense_420_kernel 
discriminator_dense_420_bias
identity??#autoencoder/StatefulPartitionedCall?%discriminator/StatefulPartitionedCall?
#autoencoder/StatefulPartitionedCallStatefulPartitionedCallinputsautoencoder_dense_412_kernelautoencoder_dense_412_biasautoencoder_dense_413_kernelautoencoder_dense_413_biasautoencoder_dense_414_kernelautoencoder_dense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609889482%
#autoencoder/StatefulPartitionedCall?
%discriminator/StatefulPartitionedCallStatefulPartitionedCall,autoencoder/StatefulPartitionedCall:output:0discriminator_dense_415_kerneldiscriminator_dense_415_biasdiscriminator_dense_416_kerneldiscriminator_dense_416_biasdiscriminator_dense_417_kerneldiscriminator_dense_417_biasdiscriminator_dense_418_kerneldiscriminator_dense_418_biasdiscriminator_dense_419_kerneldiscriminator_dense_419_biasdiscriminator_dense_420_kerneldiscriminator_dense_420_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_discriminator_layer_call_and_return_conditional_losses_609888052'
%discriminator/StatefulPartitionedCall?
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0$^autoencoder/StatefulPartitionedCall&^discriminator/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::2J
#autoencoder/StatefulPartitionedCall#autoencoder/StatefulPartitionedCall2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_415_layer_call_and_return_conditional_losses_60988532

inputs-
)tensordot_readvariableop_dense_415_kernel)
%biasadd_readvariableop_dense_415_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_415_kernel* 
_output_shapes
:
??*
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
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_415_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_dense_418_layer_call_and_return_conditional_losses_60990769

inputs-
)tensordot_readvariableop_dense_418_kernel)
%biasadd_readvariableop_dense_418_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp)tensordot_readvariableop_dense_418_kernel* 
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
:??????????2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_dense_418_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?p
?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60989032

inputs7
3dense_412_tensordot_readvariableop_dense_412_kernel3
/dense_412_biasadd_readvariableop_dense_412_bias7
3dense_413_tensordot_readvariableop_dense_413_kernel3
/dense_413_biasadd_readvariableop_dense_413_bias7
3dense_414_tensordot_readvariableop_dense_414_kernel3
/dense_414_biasadd_readvariableop_dense_414_bias
identity?? dense_412/BiasAdd/ReadVariableOp?"dense_412/Tensordot/ReadVariableOp? dense_413/BiasAdd/ReadVariableOp?"dense_413/Tensordot/ReadVariableOp? dense_414/BiasAdd/ReadVariableOp?"dense_414/Tensordot/ReadVariableOp?
"dense_412/Tensordot/ReadVariableOpReadVariableOp3dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype02$
"dense_412/Tensordot/ReadVariableOp~
dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_412/Tensordot/axes?
dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_412/Tensordot/freel
dense_412/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_412/Tensordot/Shape?
!dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/GatherV2/axis?
dense_412/Tensordot/GatherV2GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/free:output:0*dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_412/Tensordot/GatherV2?
#dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_412/Tensordot/GatherV2_1/axis?
dense_412/Tensordot/GatherV2_1GatherV2"dense_412/Tensordot/Shape:output:0!dense_412/Tensordot/axes:output:0,dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_412/Tensordot/GatherV2_1?
dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const?
dense_412/Tensordot/ProdProd%dense_412/Tensordot/GatherV2:output:0"dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod?
dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_1?
dense_412/Tensordot/Prod_1Prod'dense_412/Tensordot/GatherV2_1:output:0$dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_412/Tensordot/Prod_1?
dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_412/Tensordot/concat/axis?
dense_412/Tensordot/concatConcatV2!dense_412/Tensordot/free:output:0!dense_412/Tensordot/axes:output:0(dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat?
dense_412/Tensordot/stackPack!dense_412/Tensordot/Prod:output:0#dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/stack?
dense_412/Tensordot/transpose	Transposeinputs#dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_412/Tensordot/transpose?
dense_412/Tensordot/ReshapeReshape!dense_412/Tensordot/transpose:y:0"dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_412/Tensordot/Reshape?
dense_412/Tensordot/MatMulMatMul$dense_412/Tensordot/Reshape:output:0*dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_412/Tensordot/MatMul?
dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_412/Tensordot/Const_2?
!dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_412/Tensordot/concat_1/axis?
dense_412/Tensordot/concat_1ConcatV2%dense_412/Tensordot/GatherV2:output:0$dense_412/Tensordot/Const_2:output:0*dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_412/Tensordot/concat_1?
dense_412/TensordotReshape$dense_412/Tensordot/MatMul:product:0%dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tensordot?
 dense_412/BiasAdd/ReadVariableOpReadVariableOp/dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02"
 dense_412/BiasAdd/ReadVariableOp?
dense_412/BiasAddBiasAdddense_412/Tensordot:output:0(dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_412/BiasAddz
dense_412/TanhTanhdense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_412/Tanh?
"dense_413/Tensordot/ReadVariableOpReadVariableOp3dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype02$
"dense_413/Tensordot/ReadVariableOp~
dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_413/Tensordot/axes?
dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_413/Tensordot/freex
dense_413/Tensordot/ShapeShapedense_412/Tanh:y:0*
T0*
_output_shapes
:2
dense_413/Tensordot/Shape?
!dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/GatherV2/axis?
dense_413/Tensordot/GatherV2GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/free:output:0*dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_413/Tensordot/GatherV2?
#dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_413/Tensordot/GatherV2_1/axis?
dense_413/Tensordot/GatherV2_1GatherV2"dense_413/Tensordot/Shape:output:0!dense_413/Tensordot/axes:output:0,dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_413/Tensordot/GatherV2_1?
dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const?
dense_413/Tensordot/ProdProd%dense_413/Tensordot/GatherV2:output:0"dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod?
dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_413/Tensordot/Const_1?
dense_413/Tensordot/Prod_1Prod'dense_413/Tensordot/GatherV2_1:output:0$dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_413/Tensordot/Prod_1?
dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_413/Tensordot/concat/axis?
dense_413/Tensordot/concatConcatV2!dense_413/Tensordot/free:output:0!dense_413/Tensordot/axes:output:0(dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat?
dense_413/Tensordot/stackPack!dense_413/Tensordot/Prod:output:0#dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/stack?
dense_413/Tensordot/transpose	Transposedense_412/Tanh:y:0#dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_413/Tensordot/transpose?
dense_413/Tensordot/ReshapeReshape!dense_413/Tensordot/transpose:y:0"dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_413/Tensordot/Reshape?
dense_413/Tensordot/MatMulMatMul$dense_413/Tensordot/Reshape:output:0*dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_413/Tensordot/MatMul?
dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_413/Tensordot/Const_2?
!dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_413/Tensordot/concat_1/axis?
dense_413/Tensordot/concat_1ConcatV2%dense_413/Tensordot/GatherV2:output:0$dense_413/Tensordot/Const_2:output:0*dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_413/Tensordot/concat_1?
dense_413/TensordotReshape$dense_413/Tensordot/MatMul:product:0%dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tensordot?
 dense_413/BiasAdd/ReadVariableOpReadVariableOp/dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02"
 dense_413/BiasAdd/ReadVariableOp?
dense_413/BiasAddBiasAdddense_413/Tensordot:output:0(dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_413/BiasAdd{
dense_413/TanhTanhdense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_413/Tanh?
"dense_414/Tensordot/ReadVariableOpReadVariableOp3dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype02$
"dense_414/Tensordot/ReadVariableOp~
dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_414/Tensordot/axes?
dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_414/Tensordot/freex
dense_414/Tensordot/ShapeShapedense_413/Tanh:y:0*
T0*
_output_shapes
:2
dense_414/Tensordot/Shape?
!dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/GatherV2/axis?
dense_414/Tensordot/GatherV2GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/free:output:0*dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_414/Tensordot/GatherV2?
#dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_414/Tensordot/GatherV2_1/axis?
dense_414/Tensordot/GatherV2_1GatherV2"dense_414/Tensordot/Shape:output:0!dense_414/Tensordot/axes:output:0,dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_414/Tensordot/GatherV2_1?
dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const?
dense_414/Tensordot/ProdProd%dense_414/Tensordot/GatherV2:output:0"dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod?
dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_414/Tensordot/Const_1?
dense_414/Tensordot/Prod_1Prod'dense_414/Tensordot/GatherV2_1:output:0$dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_414/Tensordot/Prod_1?
dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_414/Tensordot/concat/axis?
dense_414/Tensordot/concatConcatV2!dense_414/Tensordot/free:output:0!dense_414/Tensordot/axes:output:0(dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat?
dense_414/Tensordot/stackPack!dense_414/Tensordot/Prod:output:0#dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/stack?
dense_414/Tensordot/transpose	Transposedense_413/Tanh:y:0#dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot/transpose?
dense_414/Tensordot/ReshapeReshape!dense_414/Tensordot/transpose:y:0"dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_414/Tensordot/Reshape?
dense_414/Tensordot/MatMulMatMul$dense_414/Tensordot/Reshape:output:0*dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_414/Tensordot/MatMul?
dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_414/Tensordot/Const_2?
!dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_414/Tensordot/concat_1/axis?
dense_414/Tensordot/concat_1ConcatV2%dense_414/Tensordot/GatherV2:output:0$dense_414/Tensordot/Const_2:output:0*dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_414/Tensordot/concat_1?
dense_414/TensordotReshape$dense_414/Tensordot/MatMul:product:0%dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_414/Tensordot?
 dense_414/BiasAdd/ReadVariableOpReadVariableOp/dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02"
 dense_414/BiasAdd/ReadVariableOp?
dense_414/BiasAddBiasAdddense_414/Tensordot:output:0(dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_414/BiasAdd?
IdentityIdentitydense_414/BiasAdd:output:0!^dense_412/BiasAdd/ReadVariableOp#^dense_412/Tensordot/ReadVariableOp!^dense_413/BiasAdd/ReadVariableOp#^dense_413/Tensordot/ReadVariableOp!^dense_414/BiasAdd/ReadVariableOp#^dense_414/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2D
 dense_412/BiasAdd/ReadVariableOp dense_412/BiasAdd/ReadVariableOp2H
"dense_412/Tensordot/ReadVariableOp"dense_412/Tensordot/ReadVariableOp2D
 dense_413/BiasAdd/ReadVariableOp dense_413/BiasAdd/ReadVariableOp2H
"dense_413/Tensordot/ReadVariableOp"dense_413/Tensordot/ReadVariableOp2D
 dense_414/BiasAdd/ReadVariableOp dense_414/BiasAdd/ReadVariableOp2H
"dense_414/Tensordot/ReadVariableOp"dense_414/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_60988443
input_11
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11dense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609884342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
input_11
??
?
G__inference_model_960_layer_call_and_return_conditional_losses_60989490

inputsC
?autoencoder_dense_412_tensordot_readvariableop_dense_412_kernel?
;autoencoder_dense_412_biasadd_readvariableop_dense_412_biasC
?autoencoder_dense_413_tensordot_readvariableop_dense_413_kernel?
;autoencoder_dense_413_biasadd_readvariableop_dense_413_biasC
?autoencoder_dense_414_tensordot_readvariableop_dense_414_kernel?
;autoencoder_dense_414_biasadd_readvariableop_dense_414_biasE
Adiscriminator_dense_415_tensordot_readvariableop_dense_415_kernelA
=discriminator_dense_415_biasadd_readvariableop_dense_415_biasE
Adiscriminator_dense_416_tensordot_readvariableop_dense_416_kernelA
=discriminator_dense_416_biasadd_readvariableop_dense_416_biasE
Adiscriminator_dense_417_tensordot_readvariableop_dense_417_kernelA
=discriminator_dense_417_biasadd_readvariableop_dense_417_biasE
Adiscriminator_dense_418_tensordot_readvariableop_dense_418_kernelA
=discriminator_dense_418_biasadd_readvariableop_dense_418_biasE
Adiscriminator_dense_419_tensordot_readvariableop_dense_419_kernelA
=discriminator_dense_419_biasadd_readvariableop_dense_419_biasB
>discriminator_dense_420_matmul_readvariableop_dense_420_kernelA
=discriminator_dense_420_biasadd_readvariableop_dense_420_bias
identity??,autoencoder/dense_412/BiasAdd/ReadVariableOp?.autoencoder/dense_412/Tensordot/ReadVariableOp?,autoencoder/dense_413/BiasAdd/ReadVariableOp?.autoencoder/dense_413/Tensordot/ReadVariableOp?,autoencoder/dense_414/BiasAdd/ReadVariableOp?.autoencoder/dense_414/Tensordot/ReadVariableOp?.discriminator/dense_415/BiasAdd/ReadVariableOp?0discriminator/dense_415/Tensordot/ReadVariableOp?.discriminator/dense_416/BiasAdd/ReadVariableOp?0discriminator/dense_416/Tensordot/ReadVariableOp?.discriminator/dense_417/BiasAdd/ReadVariableOp?0discriminator/dense_417/Tensordot/ReadVariableOp?.discriminator/dense_418/BiasAdd/ReadVariableOp?0discriminator/dense_418/Tensordot/ReadVariableOp?.discriminator/dense_419/BiasAdd/ReadVariableOp?0discriminator/dense_419/Tensordot/ReadVariableOp?.discriminator/dense_420/BiasAdd/ReadVariableOp?-discriminator/dense_420/MatMul/ReadVariableOp?
.autoencoder/dense_412/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_412_tensordot_readvariableop_dense_412_kernel*
_output_shapes
:	? *
dtype020
.autoencoder/dense_412/Tensordot/ReadVariableOp?
$autoencoder/dense_412/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_412/Tensordot/axes?
$autoencoder/dense_412/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_412/Tensordot/free?
%autoencoder/dense_412/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2'
%autoencoder/dense_412/Tensordot/Shape?
-autoencoder/dense_412/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_412/Tensordot/GatherV2/axis?
(autoencoder/dense_412/Tensordot/GatherV2GatherV2.autoencoder/dense_412/Tensordot/Shape:output:0-autoencoder/dense_412/Tensordot/free:output:06autoencoder/dense_412/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_412/Tensordot/GatherV2?
/autoencoder/dense_412/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_412/Tensordot/GatherV2_1/axis?
*autoencoder/dense_412/Tensordot/GatherV2_1GatherV2.autoencoder/dense_412/Tensordot/Shape:output:0-autoencoder/dense_412/Tensordot/axes:output:08autoencoder/dense_412/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_412/Tensordot/GatherV2_1?
%autoencoder/dense_412/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_412/Tensordot/Const?
$autoencoder/dense_412/Tensordot/ProdProd1autoencoder/dense_412/Tensordot/GatherV2:output:0.autoencoder/dense_412/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_412/Tensordot/Prod?
'autoencoder/dense_412/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_412/Tensordot/Const_1?
&autoencoder/dense_412/Tensordot/Prod_1Prod3autoencoder/dense_412/Tensordot/GatherV2_1:output:00autoencoder/dense_412/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_412/Tensordot/Prod_1?
+autoencoder/dense_412/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_412/Tensordot/concat/axis?
&autoencoder/dense_412/Tensordot/concatConcatV2-autoencoder/dense_412/Tensordot/free:output:0-autoencoder/dense_412/Tensordot/axes:output:04autoencoder/dense_412/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_412/Tensordot/concat?
%autoencoder/dense_412/Tensordot/stackPack-autoencoder/dense_412/Tensordot/Prod:output:0/autoencoder/dense_412/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_412/Tensordot/stack?
)autoencoder/dense_412/Tensordot/transpose	Transposeinputs/autoencoder/dense_412/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)autoencoder/dense_412/Tensordot/transpose?
'autoencoder/dense_412/Tensordot/ReshapeReshape-autoencoder/dense_412/Tensordot/transpose:y:0.autoencoder/dense_412/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_412/Tensordot/Reshape?
&autoencoder/dense_412/Tensordot/MatMulMatMul0autoencoder/dense_412/Tensordot/Reshape:output:06autoencoder/dense_412/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&autoencoder/dense_412/Tensordot/MatMul?
'autoencoder/dense_412/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_412/Tensordot/Const_2?
-autoencoder/dense_412/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_412/Tensordot/concat_1/axis?
(autoencoder/dense_412/Tensordot/concat_1ConcatV21autoencoder/dense_412/Tensordot/GatherV2:output:00autoencoder/dense_412/Tensordot/Const_2:output:06autoencoder/dense_412/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_412/Tensordot/concat_1?
autoencoder/dense_412/TensordotReshape0autoencoder/dense_412/Tensordot/MatMul:product:01autoencoder/dense_412/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2!
autoencoder/dense_412/Tensordot?
,autoencoder/dense_412/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_412_biasadd_readvariableop_dense_412_bias*
_output_shapes
: *
dtype02.
,autoencoder/dense_412/BiasAdd/ReadVariableOp?
autoencoder/dense_412/BiasAddBiasAdd(autoencoder/dense_412/Tensordot:output:04autoencoder/dense_412/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
autoencoder/dense_412/BiasAdd?
autoencoder/dense_412/TanhTanh&autoencoder/dense_412/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
autoencoder/dense_412/Tanh?
.autoencoder/dense_413/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_413_tensordot_readvariableop_dense_413_kernel*
_output_shapes
:	 ?*
dtype020
.autoencoder/dense_413/Tensordot/ReadVariableOp?
$autoencoder/dense_413/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_413/Tensordot/axes?
$autoencoder/dense_413/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_413/Tensordot/free?
%autoencoder/dense_413/Tensordot/ShapeShapeautoencoder/dense_412/Tanh:y:0*
T0*
_output_shapes
:2'
%autoencoder/dense_413/Tensordot/Shape?
-autoencoder/dense_413/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_413/Tensordot/GatherV2/axis?
(autoencoder/dense_413/Tensordot/GatherV2GatherV2.autoencoder/dense_413/Tensordot/Shape:output:0-autoencoder/dense_413/Tensordot/free:output:06autoencoder/dense_413/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_413/Tensordot/GatherV2?
/autoencoder/dense_413/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_413/Tensordot/GatherV2_1/axis?
*autoencoder/dense_413/Tensordot/GatherV2_1GatherV2.autoencoder/dense_413/Tensordot/Shape:output:0-autoencoder/dense_413/Tensordot/axes:output:08autoencoder/dense_413/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_413/Tensordot/GatherV2_1?
%autoencoder/dense_413/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_413/Tensordot/Const?
$autoencoder/dense_413/Tensordot/ProdProd1autoencoder/dense_413/Tensordot/GatherV2:output:0.autoencoder/dense_413/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_413/Tensordot/Prod?
'autoencoder/dense_413/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_413/Tensordot/Const_1?
&autoencoder/dense_413/Tensordot/Prod_1Prod3autoencoder/dense_413/Tensordot/GatherV2_1:output:00autoencoder/dense_413/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_413/Tensordot/Prod_1?
+autoencoder/dense_413/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_413/Tensordot/concat/axis?
&autoencoder/dense_413/Tensordot/concatConcatV2-autoencoder/dense_413/Tensordot/free:output:0-autoencoder/dense_413/Tensordot/axes:output:04autoencoder/dense_413/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_413/Tensordot/concat?
%autoencoder/dense_413/Tensordot/stackPack-autoencoder/dense_413/Tensordot/Prod:output:0/autoencoder/dense_413/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_413/Tensordot/stack?
)autoencoder/dense_413/Tensordot/transpose	Transposeautoencoder/dense_412/Tanh:y:0/autoencoder/dense_413/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2+
)autoencoder/dense_413/Tensordot/transpose?
'autoencoder/dense_413/Tensordot/ReshapeReshape-autoencoder/dense_413/Tensordot/transpose:y:0.autoencoder/dense_413/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_413/Tensordot/Reshape?
&autoencoder/dense_413/Tensordot/MatMulMatMul0autoencoder/dense_413/Tensordot/Reshape:output:06autoencoder/dense_413/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/dense_413/Tensordot/MatMul?
'autoencoder/dense_413/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'autoencoder/dense_413/Tensordot/Const_2?
-autoencoder/dense_413/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_413/Tensordot/concat_1/axis?
(autoencoder/dense_413/Tensordot/concat_1ConcatV21autoencoder/dense_413/Tensordot/GatherV2:output:00autoencoder/dense_413/Tensordot/Const_2:output:06autoencoder/dense_413/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_413/Tensordot/concat_1?
autoencoder/dense_413/TensordotReshape0autoencoder/dense_413/Tensordot/MatMul:product:01autoencoder/dense_413/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
autoencoder/dense_413/Tensordot?
,autoencoder/dense_413/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_413_biasadd_readvariableop_dense_413_bias*
_output_shapes	
:?*
dtype02.
,autoencoder/dense_413/BiasAdd/ReadVariableOp?
autoencoder/dense_413/BiasAddBiasAdd(autoencoder/dense_413/Tensordot:output:04autoencoder/dense_413/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_413/BiasAdd?
autoencoder/dense_413/TanhTanh&autoencoder/dense_413/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_413/Tanh?
.autoencoder/dense_414/Tensordot/ReadVariableOpReadVariableOp?autoencoder_dense_414_tensordot_readvariableop_dense_414_kernel* 
_output_shapes
:
??*
dtype020
.autoencoder/dense_414/Tensordot/ReadVariableOp?
$autoencoder/dense_414/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/dense_414/Tensordot/axes?
$autoencoder/dense_414/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$autoencoder/dense_414/Tensordot/free?
%autoencoder/dense_414/Tensordot/ShapeShapeautoencoder/dense_413/Tanh:y:0*
T0*
_output_shapes
:2'
%autoencoder/dense_414/Tensordot/Shape?
-autoencoder/dense_414/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_414/Tensordot/GatherV2/axis?
(autoencoder/dense_414/Tensordot/GatherV2GatherV2.autoencoder/dense_414/Tensordot/Shape:output:0-autoencoder/dense_414/Tensordot/free:output:06autoencoder/dense_414/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(autoencoder/dense_414/Tensordot/GatherV2?
/autoencoder/dense_414/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/dense_414/Tensordot/GatherV2_1/axis?
*autoencoder/dense_414/Tensordot/GatherV2_1GatherV2.autoencoder/dense_414/Tensordot/Shape:output:0-autoencoder/dense_414/Tensordot/axes:output:08autoencoder/dense_414/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*autoencoder/dense_414/Tensordot/GatherV2_1?
%autoencoder/dense_414/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%autoencoder/dense_414/Tensordot/Const?
$autoencoder/dense_414/Tensordot/ProdProd1autoencoder/dense_414/Tensordot/GatherV2:output:0.autoencoder/dense_414/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$autoencoder/dense_414/Tensordot/Prod?
'autoencoder/dense_414/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'autoencoder/dense_414/Tensordot/Const_1?
&autoencoder/dense_414/Tensordot/Prod_1Prod3autoencoder/dense_414/Tensordot/GatherV2_1:output:00autoencoder/dense_414/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&autoencoder/dense_414/Tensordot/Prod_1?
+autoencoder/dense_414/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/dense_414/Tensordot/concat/axis?
&autoencoder/dense_414/Tensordot/concatConcatV2-autoencoder/dense_414/Tensordot/free:output:0-autoencoder/dense_414/Tensordot/axes:output:04autoencoder/dense_414/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&autoencoder/dense_414/Tensordot/concat?
%autoencoder/dense_414/Tensordot/stackPack-autoencoder/dense_414/Tensordot/Prod:output:0/autoencoder/dense_414/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/dense_414/Tensordot/stack?
)autoencoder/dense_414/Tensordot/transpose	Transposeautoencoder/dense_413/Tanh:y:0/autoencoder/dense_414/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)autoencoder/dense_414/Tensordot/transpose?
'autoencoder/dense_414/Tensordot/ReshapeReshape-autoencoder/dense_414/Tensordot/transpose:y:0.autoencoder/dense_414/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'autoencoder/dense_414/Tensordot/Reshape?
&autoencoder/dense_414/Tensordot/MatMulMatMul0autoencoder/dense_414/Tensordot/Reshape:output:06autoencoder/dense_414/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&autoencoder/dense_414/Tensordot/MatMul?
'autoencoder/dense_414/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'autoencoder/dense_414/Tensordot/Const_2?
-autoencoder/dense_414/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-autoencoder/dense_414/Tensordot/concat_1/axis?
(autoencoder/dense_414/Tensordot/concat_1ConcatV21autoencoder/dense_414/Tensordot/GatherV2:output:00autoencoder/dense_414/Tensordot/Const_2:output:06autoencoder/dense_414/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(autoencoder/dense_414/Tensordot/concat_1?
autoencoder/dense_414/TensordotReshape0autoencoder/dense_414/Tensordot/MatMul:product:01autoencoder/dense_414/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
autoencoder/dense_414/Tensordot?
,autoencoder/dense_414/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_dense_414_biasadd_readvariableop_dense_414_bias*
_output_shapes	
:?*
dtype02.
,autoencoder/dense_414/BiasAdd/ReadVariableOp?
autoencoder/dense_414/BiasAddBiasAdd(autoencoder/dense_414/Tensordot:output:04autoencoder/dense_414/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
autoencoder/dense_414/BiasAdd?
#discriminator/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#discriminator/dropout/dropout/Const?
!discriminator/dropout/dropout/MulMul&autoencoder/dense_414/BiasAdd:output:0,discriminator/dropout/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dropout/dropout/Mul?
#discriminator/dropout/dropout/ShapeShape&autoencoder/dense_414/BiasAdd:output:0*
T0*
_output_shapes
:2%
#discriminator/dropout/dropout/Shape?
:discriminator/dropout/dropout/random_uniform/RandomUniformRandomUniform,discriminator/dropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02<
:discriminator/dropout/dropout/random_uniform/RandomUniform?
,discriminator/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,discriminator/dropout/dropout/GreaterEqual/y?
*discriminator/dropout/dropout/GreaterEqualGreaterEqualCdiscriminator/dropout/dropout/random_uniform/RandomUniform:output:05discriminator/dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2,
*discriminator/dropout/dropout/GreaterEqual?
"discriminator/dropout/dropout/CastCast.discriminator/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2$
"discriminator/dropout/dropout/Cast?
#discriminator/dropout/dropout/Mul_1Mul%discriminator/dropout/dropout/Mul:z:0&discriminator/dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2%
#discriminator/dropout/dropout/Mul_1?
0discriminator/dense_415/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_415_tensordot_readvariableop_dense_415_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_415/Tensordot/ReadVariableOp?
&discriminator/dense_415/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_415/Tensordot/axes?
&discriminator/dense_415/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_415/Tensordot/free?
'discriminator/dense_415/Tensordot/ShapeShape'discriminator/dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2)
'discriminator/dense_415/Tensordot/Shape?
/discriminator/dense_415/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_415/Tensordot/GatherV2/axis?
*discriminator/dense_415/Tensordot/GatherV2GatherV20discriminator/dense_415/Tensordot/Shape:output:0/discriminator/dense_415/Tensordot/free:output:08discriminator/dense_415/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_415/Tensordot/GatherV2?
1discriminator/dense_415/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_415/Tensordot/GatherV2_1/axis?
,discriminator/dense_415/Tensordot/GatherV2_1GatherV20discriminator/dense_415/Tensordot/Shape:output:0/discriminator/dense_415/Tensordot/axes:output:0:discriminator/dense_415/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_415/Tensordot/GatherV2_1?
'discriminator/dense_415/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_415/Tensordot/Const?
&discriminator/dense_415/Tensordot/ProdProd3discriminator/dense_415/Tensordot/GatherV2:output:00discriminator/dense_415/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_415/Tensordot/Prod?
)discriminator/dense_415/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_415/Tensordot/Const_1?
(discriminator/dense_415/Tensordot/Prod_1Prod5discriminator/dense_415/Tensordot/GatherV2_1:output:02discriminator/dense_415/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_415/Tensordot/Prod_1?
-discriminator/dense_415/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_415/Tensordot/concat/axis?
(discriminator/dense_415/Tensordot/concatConcatV2/discriminator/dense_415/Tensordot/free:output:0/discriminator/dense_415/Tensordot/axes:output:06discriminator/dense_415/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_415/Tensordot/concat?
'discriminator/dense_415/Tensordot/stackPack/discriminator/dense_415/Tensordot/Prod:output:01discriminator/dense_415/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_415/Tensordot/stack?
+discriminator/dense_415/Tensordot/transpose	Transpose'discriminator/dropout/dropout/Mul_1:z:01discriminator/dense_415/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_415/Tensordot/transpose?
)discriminator/dense_415/Tensordot/ReshapeReshape/discriminator/dense_415/Tensordot/transpose:y:00discriminator/dense_415/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_415/Tensordot/Reshape?
(discriminator/dense_415/Tensordot/MatMulMatMul2discriminator/dense_415/Tensordot/Reshape:output:08discriminator/dense_415/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_415/Tensordot/MatMul?
)discriminator/dense_415/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_415/Tensordot/Const_2?
/discriminator/dense_415/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_415/Tensordot/concat_1/axis?
*discriminator/dense_415/Tensordot/concat_1ConcatV23discriminator/dense_415/Tensordot/GatherV2:output:02discriminator/dense_415/Tensordot/Const_2:output:08discriminator/dense_415/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_415/Tensordot/concat_1?
!discriminator/dense_415/TensordotReshape2discriminator/dense_415/Tensordot/MatMul:product:03discriminator/dense_415/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_415/Tensordot?
.discriminator/dense_415/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_415_biasadd_readvariableop_dense_415_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_415/BiasAdd/ReadVariableOp?
discriminator/dense_415/BiasAddBiasAdd*discriminator/dense_415/Tensordot:output:06discriminator/dense_415/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_415/BiasAdd?
discriminator/dense_415/TanhTanh(discriminator/dense_415/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_415/Tanh?
0discriminator/dense_416/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_416_tensordot_readvariableop_dense_416_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_416/Tensordot/ReadVariableOp?
&discriminator/dense_416/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_416/Tensordot/axes?
&discriminator/dense_416/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_416/Tensordot/free?
'discriminator/dense_416/Tensordot/ShapeShape discriminator/dense_415/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_416/Tensordot/Shape?
/discriminator/dense_416/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_416/Tensordot/GatherV2/axis?
*discriminator/dense_416/Tensordot/GatherV2GatherV20discriminator/dense_416/Tensordot/Shape:output:0/discriminator/dense_416/Tensordot/free:output:08discriminator/dense_416/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_416/Tensordot/GatherV2?
1discriminator/dense_416/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_416/Tensordot/GatherV2_1/axis?
,discriminator/dense_416/Tensordot/GatherV2_1GatherV20discriminator/dense_416/Tensordot/Shape:output:0/discriminator/dense_416/Tensordot/axes:output:0:discriminator/dense_416/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_416/Tensordot/GatherV2_1?
'discriminator/dense_416/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_416/Tensordot/Const?
&discriminator/dense_416/Tensordot/ProdProd3discriminator/dense_416/Tensordot/GatherV2:output:00discriminator/dense_416/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_416/Tensordot/Prod?
)discriminator/dense_416/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_416/Tensordot/Const_1?
(discriminator/dense_416/Tensordot/Prod_1Prod5discriminator/dense_416/Tensordot/GatherV2_1:output:02discriminator/dense_416/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_416/Tensordot/Prod_1?
-discriminator/dense_416/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_416/Tensordot/concat/axis?
(discriminator/dense_416/Tensordot/concatConcatV2/discriminator/dense_416/Tensordot/free:output:0/discriminator/dense_416/Tensordot/axes:output:06discriminator/dense_416/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_416/Tensordot/concat?
'discriminator/dense_416/Tensordot/stackPack/discriminator/dense_416/Tensordot/Prod:output:01discriminator/dense_416/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_416/Tensordot/stack?
+discriminator/dense_416/Tensordot/transpose	Transpose discriminator/dense_415/Tanh:y:01discriminator/dense_416/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_416/Tensordot/transpose?
)discriminator/dense_416/Tensordot/ReshapeReshape/discriminator/dense_416/Tensordot/transpose:y:00discriminator/dense_416/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_416/Tensordot/Reshape?
(discriminator/dense_416/Tensordot/MatMulMatMul2discriminator/dense_416/Tensordot/Reshape:output:08discriminator/dense_416/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_416/Tensordot/MatMul?
)discriminator/dense_416/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_416/Tensordot/Const_2?
/discriminator/dense_416/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_416/Tensordot/concat_1/axis?
*discriminator/dense_416/Tensordot/concat_1ConcatV23discriminator/dense_416/Tensordot/GatherV2:output:02discriminator/dense_416/Tensordot/Const_2:output:08discriminator/dense_416/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_416/Tensordot/concat_1?
!discriminator/dense_416/TensordotReshape2discriminator/dense_416/Tensordot/MatMul:product:03discriminator/dense_416/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_416/Tensordot?
.discriminator/dense_416/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_416_biasadd_readvariableop_dense_416_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_416/BiasAdd/ReadVariableOp?
discriminator/dense_416/BiasAddBiasAdd*discriminator/dense_416/Tensordot:output:06discriminator/dense_416/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_416/BiasAdd?
discriminator/dense_416/TanhTanh(discriminator/dense_416/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_416/Tanh?
0discriminator/dense_417/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_417_tensordot_readvariableop_dense_417_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_417/Tensordot/ReadVariableOp?
&discriminator/dense_417/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_417/Tensordot/axes?
&discriminator/dense_417/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_417/Tensordot/free?
'discriminator/dense_417/Tensordot/ShapeShape discriminator/dense_416/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_417/Tensordot/Shape?
/discriminator/dense_417/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_417/Tensordot/GatherV2/axis?
*discriminator/dense_417/Tensordot/GatherV2GatherV20discriminator/dense_417/Tensordot/Shape:output:0/discriminator/dense_417/Tensordot/free:output:08discriminator/dense_417/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_417/Tensordot/GatherV2?
1discriminator/dense_417/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_417/Tensordot/GatherV2_1/axis?
,discriminator/dense_417/Tensordot/GatherV2_1GatherV20discriminator/dense_417/Tensordot/Shape:output:0/discriminator/dense_417/Tensordot/axes:output:0:discriminator/dense_417/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_417/Tensordot/GatherV2_1?
'discriminator/dense_417/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_417/Tensordot/Const?
&discriminator/dense_417/Tensordot/ProdProd3discriminator/dense_417/Tensordot/GatherV2:output:00discriminator/dense_417/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_417/Tensordot/Prod?
)discriminator/dense_417/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_417/Tensordot/Const_1?
(discriminator/dense_417/Tensordot/Prod_1Prod5discriminator/dense_417/Tensordot/GatherV2_1:output:02discriminator/dense_417/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_417/Tensordot/Prod_1?
-discriminator/dense_417/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_417/Tensordot/concat/axis?
(discriminator/dense_417/Tensordot/concatConcatV2/discriminator/dense_417/Tensordot/free:output:0/discriminator/dense_417/Tensordot/axes:output:06discriminator/dense_417/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_417/Tensordot/concat?
'discriminator/dense_417/Tensordot/stackPack/discriminator/dense_417/Tensordot/Prod:output:01discriminator/dense_417/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_417/Tensordot/stack?
+discriminator/dense_417/Tensordot/transpose	Transpose discriminator/dense_416/Tanh:y:01discriminator/dense_417/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_417/Tensordot/transpose?
)discriminator/dense_417/Tensordot/ReshapeReshape/discriminator/dense_417/Tensordot/transpose:y:00discriminator/dense_417/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_417/Tensordot/Reshape?
(discriminator/dense_417/Tensordot/MatMulMatMul2discriminator/dense_417/Tensordot/Reshape:output:08discriminator/dense_417/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_417/Tensordot/MatMul?
)discriminator/dense_417/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_417/Tensordot/Const_2?
/discriminator/dense_417/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_417/Tensordot/concat_1/axis?
*discriminator/dense_417/Tensordot/concat_1ConcatV23discriminator/dense_417/Tensordot/GatherV2:output:02discriminator/dense_417/Tensordot/Const_2:output:08discriminator/dense_417/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_417/Tensordot/concat_1?
!discriminator/dense_417/TensordotReshape2discriminator/dense_417/Tensordot/MatMul:product:03discriminator/dense_417/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_417/Tensordot?
.discriminator/dense_417/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_417_biasadd_readvariableop_dense_417_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_417/BiasAdd/ReadVariableOp?
discriminator/dense_417/BiasAddBiasAdd*discriminator/dense_417/Tensordot:output:06discriminator/dense_417/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_417/BiasAdd?
discriminator/dense_417/TanhTanh(discriminator/dense_417/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_417/Tanh?
0discriminator/dense_418/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_418_tensordot_readvariableop_dense_418_kernel* 
_output_shapes
:
??*
dtype022
0discriminator/dense_418/Tensordot/ReadVariableOp?
&discriminator/dense_418/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_418/Tensordot/axes?
&discriminator/dense_418/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_418/Tensordot/free?
'discriminator/dense_418/Tensordot/ShapeShape discriminator/dense_417/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_418/Tensordot/Shape?
/discriminator/dense_418/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_418/Tensordot/GatherV2/axis?
*discriminator/dense_418/Tensordot/GatherV2GatherV20discriminator/dense_418/Tensordot/Shape:output:0/discriminator/dense_418/Tensordot/free:output:08discriminator/dense_418/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_418/Tensordot/GatherV2?
1discriminator/dense_418/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_418/Tensordot/GatherV2_1/axis?
,discriminator/dense_418/Tensordot/GatherV2_1GatherV20discriminator/dense_418/Tensordot/Shape:output:0/discriminator/dense_418/Tensordot/axes:output:0:discriminator/dense_418/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_418/Tensordot/GatherV2_1?
'discriminator/dense_418/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_418/Tensordot/Const?
&discriminator/dense_418/Tensordot/ProdProd3discriminator/dense_418/Tensordot/GatherV2:output:00discriminator/dense_418/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_418/Tensordot/Prod?
)discriminator/dense_418/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_418/Tensordot/Const_1?
(discriminator/dense_418/Tensordot/Prod_1Prod5discriminator/dense_418/Tensordot/GatherV2_1:output:02discriminator/dense_418/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_418/Tensordot/Prod_1?
-discriminator/dense_418/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_418/Tensordot/concat/axis?
(discriminator/dense_418/Tensordot/concatConcatV2/discriminator/dense_418/Tensordot/free:output:0/discriminator/dense_418/Tensordot/axes:output:06discriminator/dense_418/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_418/Tensordot/concat?
'discriminator/dense_418/Tensordot/stackPack/discriminator/dense_418/Tensordot/Prod:output:01discriminator/dense_418/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_418/Tensordot/stack?
+discriminator/dense_418/Tensordot/transpose	Transpose discriminator/dense_417/Tanh:y:01discriminator/dense_418/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_418/Tensordot/transpose?
)discriminator/dense_418/Tensordot/ReshapeReshape/discriminator/dense_418/Tensordot/transpose:y:00discriminator/dense_418/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_418/Tensordot/Reshape?
(discriminator/dense_418/Tensordot/MatMulMatMul2discriminator/dense_418/Tensordot/Reshape:output:08discriminator/dense_418/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(discriminator/dense_418/Tensordot/MatMul?
)discriminator/dense_418/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2+
)discriminator/dense_418/Tensordot/Const_2?
/discriminator/dense_418/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_418/Tensordot/concat_1/axis?
*discriminator/dense_418/Tensordot/concat_1ConcatV23discriminator/dense_418/Tensordot/GatherV2:output:02discriminator/dense_418/Tensordot/Const_2:output:08discriminator/dense_418/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_418/Tensordot/concat_1?
!discriminator/dense_418/TensordotReshape2discriminator/dense_418/Tensordot/MatMul:product:03discriminator/dense_418/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2#
!discriminator/dense_418/Tensordot?
.discriminator/dense_418/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_418_biasadd_readvariableop_dense_418_bias*
_output_shapes	
:?*
dtype020
.discriminator/dense_418/BiasAdd/ReadVariableOp?
discriminator/dense_418/BiasAddBiasAdd*discriminator/dense_418/Tensordot:output:06discriminator/dense_418/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2!
discriminator/dense_418/BiasAdd?
discriminator/dense_418/TanhTanh(discriminator/dense_418/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
discriminator/dense_418/Tanh?
0discriminator/dense_419/Tensordot/ReadVariableOpReadVariableOpAdiscriminator_dense_419_tensordot_readvariableop_dense_419_kernel*
_output_shapes
:	?*
dtype022
0discriminator/dense_419/Tensordot/ReadVariableOp?
&discriminator/dense_419/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&discriminator/dense_419/Tensordot/axes?
&discriminator/dense_419/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&discriminator/dense_419/Tensordot/free?
'discriminator/dense_419/Tensordot/ShapeShape discriminator/dense_418/Tanh:y:0*
T0*
_output_shapes
:2)
'discriminator/dense_419/Tensordot/Shape?
/discriminator/dense_419/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_419/Tensordot/GatherV2/axis?
*discriminator/dense_419/Tensordot/GatherV2GatherV20discriminator/dense_419/Tensordot/Shape:output:0/discriminator/dense_419/Tensordot/free:output:08discriminator/dense_419/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*discriminator/dense_419/Tensordot/GatherV2?
1discriminator/dense_419/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1discriminator/dense_419/Tensordot/GatherV2_1/axis?
,discriminator/dense_419/Tensordot/GatherV2_1GatherV20discriminator/dense_419/Tensordot/Shape:output:0/discriminator/dense_419/Tensordot/axes:output:0:discriminator/dense_419/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,discriminator/dense_419/Tensordot/GatherV2_1?
'discriminator/dense_419/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'discriminator/dense_419/Tensordot/Const?
&discriminator/dense_419/Tensordot/ProdProd3discriminator/dense_419/Tensordot/GatherV2:output:00discriminator/dense_419/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&discriminator/dense_419/Tensordot/Prod?
)discriminator/dense_419/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)discriminator/dense_419/Tensordot/Const_1?
(discriminator/dense_419/Tensordot/Prod_1Prod5discriminator/dense_419/Tensordot/GatherV2_1:output:02discriminator/dense_419/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(discriminator/dense_419/Tensordot/Prod_1?
-discriminator/dense_419/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-discriminator/dense_419/Tensordot/concat/axis?
(discriminator/dense_419/Tensordot/concatConcatV2/discriminator/dense_419/Tensordot/free:output:0/discriminator/dense_419/Tensordot/axes:output:06discriminator/dense_419/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(discriminator/dense_419/Tensordot/concat?
'discriminator/dense_419/Tensordot/stackPack/discriminator/dense_419/Tensordot/Prod:output:01discriminator/dense_419/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'discriminator/dense_419/Tensordot/stack?
+discriminator/dense_419/Tensordot/transpose	Transpose discriminator/dense_418/Tanh:y:01discriminator/dense_419/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2-
+discriminator/dense_419/Tensordot/transpose?
)discriminator/dense_419/Tensordot/ReshapeReshape/discriminator/dense_419/Tensordot/transpose:y:00discriminator/dense_419/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)discriminator/dense_419/Tensordot/Reshape?
(discriminator/dense_419/Tensordot/MatMulMatMul2discriminator/dense_419/Tensordot/Reshape:output:08discriminator/dense_419/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(discriminator/dense_419/Tensordot/MatMul?
)discriminator/dense_419/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)discriminator/dense_419/Tensordot/Const_2?
/discriminator/dense_419/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/discriminator/dense_419/Tensordot/concat_1/axis?
*discriminator/dense_419/Tensordot/concat_1ConcatV23discriminator/dense_419/Tensordot/GatherV2:output:02discriminator/dense_419/Tensordot/Const_2:output:08discriminator/dense_419/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*discriminator/dense_419/Tensordot/concat_1?
!discriminator/dense_419/TensordotReshape2discriminator/dense_419/Tensordot/MatMul:product:03discriminator/dense_419/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2#
!discriminator/dense_419/Tensordot?
.discriminator/dense_419/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_419_biasadd_readvariableop_dense_419_bias*
_output_shapes
:*
dtype020
.discriminator/dense_419/BiasAdd/ReadVariableOp?
discriminator/dense_419/BiasAddBiasAdd*discriminator/dense_419/Tensordot:output:06discriminator/dense_419/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2!
discriminator/dense_419/BiasAdd?
discriminator/dense_419/TanhTanh(discriminator/dense_419/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
discriminator/dense_419/Tanh?
discriminator/flatten_51/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2 
discriminator/flatten_51/Const?
 discriminator/flatten_51/ReshapeReshape discriminator/dense_419/Tanh:y:0'discriminator/flatten_51/Const:output:0*
T0*'
_output_shapes
:?????????2"
 discriminator/flatten_51/Reshape?
-discriminator/dense_420/MatMul/ReadVariableOpReadVariableOp>discriminator_dense_420_matmul_readvariableop_dense_420_kernel*
_output_shapes

:*
dtype02/
-discriminator/dense_420/MatMul/ReadVariableOp?
discriminator/dense_420/MatMulMatMul)discriminator/flatten_51/Reshape:output:05discriminator/dense_420/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
discriminator/dense_420/MatMul?
.discriminator/dense_420/BiasAdd/ReadVariableOpReadVariableOp=discriminator_dense_420_biasadd_readvariableop_dense_420_bias*
_output_shapes
:*
dtype020
.discriminator/dense_420/BiasAdd/ReadVariableOp?
discriminator/dense_420/BiasAddBiasAdd(discriminator/dense_420/MatMul:product:06discriminator/dense_420/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
discriminator/dense_420/BiasAdd?
discriminator/dense_420/SigmoidSigmoid(discriminator/dense_420/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2!
discriminator/dense_420/Sigmoid?
IdentityIdentity#discriminator/dense_420/Sigmoid:y:0-^autoencoder/dense_412/BiasAdd/ReadVariableOp/^autoencoder/dense_412/Tensordot/ReadVariableOp-^autoencoder/dense_413/BiasAdd/ReadVariableOp/^autoencoder/dense_413/Tensordot/ReadVariableOp-^autoencoder/dense_414/BiasAdd/ReadVariableOp/^autoencoder/dense_414/Tensordot/ReadVariableOp/^discriminator/dense_415/BiasAdd/ReadVariableOp1^discriminator/dense_415/Tensordot/ReadVariableOp/^discriminator/dense_416/BiasAdd/ReadVariableOp1^discriminator/dense_416/Tensordot/ReadVariableOp/^discriminator/dense_417/BiasAdd/ReadVariableOp1^discriminator/dense_417/Tensordot/ReadVariableOp/^discriminator/dense_418/BiasAdd/ReadVariableOp1^discriminator/dense_418/Tensordot/ReadVariableOp/^discriminator/dense_419/BiasAdd/ReadVariableOp1^discriminator/dense_419/Tensordot/ReadVariableOp/^discriminator/dense_420/BiasAdd/ReadVariableOp.^discriminator/dense_420/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:??????????::::::::::::::::::2\
,autoencoder/dense_412/BiasAdd/ReadVariableOp,autoencoder/dense_412/BiasAdd/ReadVariableOp2`
.autoencoder/dense_412/Tensordot/ReadVariableOp.autoencoder/dense_412/Tensordot/ReadVariableOp2\
,autoencoder/dense_413/BiasAdd/ReadVariableOp,autoencoder/dense_413/BiasAdd/ReadVariableOp2`
.autoencoder/dense_413/Tensordot/ReadVariableOp.autoencoder/dense_413/Tensordot/ReadVariableOp2\
,autoencoder/dense_414/BiasAdd/ReadVariableOp,autoencoder/dense_414/BiasAdd/ReadVariableOp2`
.autoencoder/dense_414/Tensordot/ReadVariableOp.autoencoder/dense_414/Tensordot/ReadVariableOp2`
.discriminator/dense_415/BiasAdd/ReadVariableOp.discriminator/dense_415/BiasAdd/ReadVariableOp2d
0discriminator/dense_415/Tensordot/ReadVariableOp0discriminator/dense_415/Tensordot/ReadVariableOp2`
.discriminator/dense_416/BiasAdd/ReadVariableOp.discriminator/dense_416/BiasAdd/ReadVariableOp2d
0discriminator/dense_416/Tensordot/ReadVariableOp0discriminator/dense_416/Tensordot/ReadVariableOp2`
.discriminator/dense_417/BiasAdd/ReadVariableOp.discriminator/dense_417/BiasAdd/ReadVariableOp2d
0discriminator/dense_417/Tensordot/ReadVariableOp0discriminator/dense_417/Tensordot/ReadVariableOp2`
.discriminator/dense_418/BiasAdd/ReadVariableOp.discriminator/dense_418/BiasAdd/ReadVariableOp2d
0discriminator/dense_418/Tensordot/ReadVariableOp0discriminator/dense_418/Tensordot/ReadVariableOp2`
.discriminator/dense_419/BiasAdd/ReadVariableOp.discriminator/dense_419/BiasAdd/ReadVariableOp2d
0discriminator/dense_419/Tensordot/ReadVariableOp0discriminator/dense_419/Tensordot/ReadVariableOp2`
.discriminator/dense_420/BiasAdd/ReadVariableOp.discriminator/dense_420/BiasAdd/ReadVariableOp2^
-discriminator/dense_420/MatMul/ReadVariableOp-discriminator/dense_420/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_autoencoder_layer_call_fn_60990145
inputs_0
dense_412_kernel
dense_412_bias
dense_413_kernel
dense_413_bias
dense_414_kernel
dense_414_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0dense_412_kerneldense_412_biasdense_413_kerneldense_413_biasdense_414_kerneldense_414_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_autoencoder_layer_call_and_return_conditional_losses_609890322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????
"
_user_specified_name
inputs/0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
B
input_116
serving_default_input_11:0??????????A
discriminator0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?i
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
?_default_save_signature"?g
_tf_keras_network?g{"class_name": "Functional", "name": "model_960", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_960", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_412", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_412", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_413", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_413", "inbound_nodes": [[["dense_412", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_414", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_414", "inbound_nodes": [[["dense_413", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["dense_414", 0, 0]]}, "name": "autoencoder", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["input_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_415", "trainable": false, "dtype": "float32", "units": 1024, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_415", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_416", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_416", "inbound_nodes": [[["dense_415", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_417", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_417", "inbound_nodes": [[["dense_416", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_418", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_418", "inbound_nodes": [[["dense_417", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_419", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_419", "inbound_nodes": [[["dense_418", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_51", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_51", "inbound_nodes": [[["dense_419", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_420", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_420", "inbound_nodes": [[["flatten_51", 0, 0, {}]]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["dense_420", 0, 0]]}, "name": "discriminator", "inbound_nodes": [[["autoencoder", 1, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["discriminator", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 513]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_960", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_412", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_412", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_413", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_413", "inbound_nodes": [[["dense_412", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_414", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_414", "inbound_nodes": [[["dense_413", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["dense_414", 0, 0]]}, "name": "autoencoder", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["input_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_415", "trainable": false, "dtype": "float32", "units": 1024, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_415", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_416", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_416", "inbound_nodes": [[["dense_415", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_417", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_417", "inbound_nodes": [[["dense_416", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_418", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_418", "inbound_nodes": [[["dense_417", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_419", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_419", "inbound_nodes": [[["dense_418", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_51", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_51", "inbound_nodes": [[["dense_419", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_420", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_420", "inbound_nodes": [[["flatten_51", 0, 0, {}]]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["dense_420", 0, 0]]}, "name": "discriminator", "inbound_nodes": [[["autoencoder", 1, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["discriminator", 1, 0]]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_11", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}}
?&
layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?#
_tf_keras_network?#{"class_name": "Functional", "name": "autoencoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_412", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_412", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_413", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_413", "inbound_nodes": [[["dense_412", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_414", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_414", "inbound_nodes": [[["dense_413", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["dense_414", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 513]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_412", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_412", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_413", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_413", "inbound_nodes": [[["dense_412", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_414", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_414", "inbound_nodes": [[["dense_413", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["dense_414", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?E
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
layer_with_weights-5
layer-8
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?A
_tf_keras_network?A{"class_name": "Functional", "name": "discriminator", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["input_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_415", "trainable": false, "dtype": "float32", "units": 1024, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_415", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_416", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_416", "inbound_nodes": [[["dense_415", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_417", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_417", "inbound_nodes": [[["dense_416", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_418", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_418", "inbound_nodes": [[["dense_417", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_419", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_419", "inbound_nodes": [[["dense_418", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_51", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_51", "inbound_nodes": [[["dense_419", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_420", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_420", "inbound_nodes": [[["flatten_51", 0, 0, {}]]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["dense_420", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 513]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["input_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_415", "trainable": false, "dtype": "float32", "units": 1024, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_415", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_416", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_416", "inbound_nodes": [[["dense_415", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_417", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_417", "inbound_nodes": [[["dense_416", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_418", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_418", "inbound_nodes": [[["dense_417", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_419", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_419", "inbound_nodes": [[["dense_418", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_51", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_51", "inbound_nodes": [[["dense_419", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_420", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_420", "inbound_nodes": [[["flatten_51", 0, 0, {}]]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["dense_420", 0, 0]]}}, "training_config": {"loss": "bce", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
415
516
617"
trackable_list_wrapper
?
regularization_losses
trainable_variables

7layers
8layer_regularization_losses
9metrics
:layer_metrics
;non_trainable_variables
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
<regularization_losses
=trainable_variables
>	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_412", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_412", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 513}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 513]}}
?

'kernel
(bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_413", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_413", "trainable": true, "dtype": "float32", "units": 513, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 32]}}
?

)kernel
*bias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_414", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_414", "trainable": true, "dtype": "float32", "units": 513, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 513}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 513]}}
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
regularization_losses
trainable_variables

Hlayers
Ilayer_regularization_losses
Jmetrics
Klayer_metrics
Lnon_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_12", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 513]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}}
?
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

+kernel
,bias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_415", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_415", "trainable": false, "dtype": "float32", "units": 1024, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 513}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 513]}}
?

-kernel
.bias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_416", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_416", "trainable": false, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1024]}}
?

/kernel
0bias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_417", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_417", "trainable": false, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 512]}}
?

1kernel
2bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_418", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_418", "trainable": false, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 256]}}
?

3kernel
4bias
aregularization_losses
btrainable_variables
c	variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_419", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_419", "trainable": false, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 128]}}
?
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_51", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_51", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

5kernel
6bias
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_420", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_420", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
?
miter

nbeta_1

obeta_2
	pdecay
qlearning_rate+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
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
510
611"
trackable_list_wrapper
?
regularization_losses
trainable_variables

rlayers
slayer_regularization_losses
tmetrics
ulayer_metrics
vnon_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	 (2training_318/Adam/iter
":  (2training_318/Adam/beta_1
":  (2training_318/Adam/beta_2
!: (2training_318/Adam/decay
):' (2training_318/Adam/learning_rate
#:!	? 2dense_412/kernel
: 2dense_412/bias
#:!	 ?2dense_413/kernel
:?2dense_413/bias
$:"
??2dense_414/kernel
:?2dense_414/bias
$:"
??2dense_415/kernel
:?2dense_415/bias
$:"
??2dense_416/kernel
:?2dense_416/bias
$:"
??2dense_417/kernel
:?2dense_417/bias
$:"
??2dense_418/kernel
:?2dense_418/bias
#:!	?2dense_419/kernel
:2dense_419/bias
": 2dense_420/kernel
:2dense_420/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
'
w0"
trackable_list_wrapper
 "
trackable_dict_wrapper
v
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
510
611"
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
<regularization_losses
=trainable_variables
xlayer_regularization_losses

ylayers
zmetrics
{layer_metrics
|non_trainable_variables
>	variables
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
@regularization_losses
Atrainable_variables
}layer_regularization_losses

~layers
metrics
?layer_metrics
?non_trainable_variables
B	variables
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
Dregularization_losses
Etrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
F	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
<
0

1
2
3"
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
Mregularization_losses
Ntrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
O	variables
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
Qregularization_losses
Rtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
S	variables
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
Uregularization_losses
Vtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
W	variables
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
Yregularization_losses
Ztrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
[	variables
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
]regularization_losses
^trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
_	variables
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
aregularization_losses
btrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
c	variables
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
eregularization_losses
ftrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
g	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
iregularization_losses
jtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
?layer_metrics
?non_trainable_variables
k	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	 (2training_130/Adam/iter
":  (2training_130/Adam/beta_1
":  (2training_130/Adam/beta_2
!: (2training_130/Adam/decay
):' (2training_130/Adam/learning_rate
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
v
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
510
611"
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
.
30
41"
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
50
61"
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2	total_329
:  (2	count_329
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2	total_139
:  (2	count_139
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2	total_140
:  (2	count_140
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
5:3	? 2$training_318/Adam/dense_412/kernel/m
.:, 2"training_318/Adam/dense_412/bias/m
5:3	 ?2$training_318/Adam/dense_413/kernel/m
/:-?2"training_318/Adam/dense_413/bias/m
6:4
??2$training_318/Adam/dense_414/kernel/m
/:-?2"training_318/Adam/dense_414/bias/m
5:3	? 2$training_318/Adam/dense_412/kernel/v
.:, 2"training_318/Adam/dense_412/bias/v
5:3	 ?2$training_318/Adam/dense_413/kernel/v
/:-?2"training_318/Adam/dense_413/bias/v
6:4
??2$training_318/Adam/dense_414/kernel/v
/:-?2"training_318/Adam/dense_414/bias/v
6:4
??2$training_130/Adam/dense_415/kernel/m
/:-?2"training_130/Adam/dense_415/bias/m
6:4
??2$training_130/Adam/dense_416/kernel/m
/:-?2"training_130/Adam/dense_416/bias/m
6:4
??2$training_130/Adam/dense_417/kernel/m
/:-?2"training_130/Adam/dense_417/bias/m
6:4
??2$training_130/Adam/dense_418/kernel/m
/:-?2"training_130/Adam/dense_418/bias/m
5:3	?2$training_130/Adam/dense_419/kernel/m
.:,2"training_130/Adam/dense_419/bias/m
4:22$training_130/Adam/dense_420/kernel/m
.:,2"training_130/Adam/dense_420/bias/m
6:4
??2$training_130/Adam/dense_415/kernel/v
/:-?2"training_130/Adam/dense_415/bias/v
6:4
??2$training_130/Adam/dense_416/kernel/v
/:-?2"training_130/Adam/dense_416/bias/v
6:4
??2$training_130/Adam/dense_417/kernel/v
/:-?2"training_130/Adam/dense_417/bias/v
6:4
??2$training_130/Adam/dense_418/kernel/v
/:-?2"training_130/Adam/dense_418/bias/v
5:3	?2$training_130/Adam/dense_419/kernel/v
.:,2"training_130/Adam/dense_419/bias/v
4:22$training_130/Adam/dense_420/kernel/v
.:,2"training_130/Adam/dense_420/bias/v
?2?
G__inference_model_960_layer_call_and_return_conditional_losses_60989719
G__inference_model_960_layer_call_and_return_conditional_losses_60989490
G__inference_model_960_layer_call_and_return_conditional_losses_60989110
G__inference_model_960_layer_call_and_return_conditional_losses_60989134?
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
,__inference_model_960_layer_call_fn_60989742
,__inference_model_960_layer_call_fn_60989765
,__inference_model_960_layer_call_fn_60989229
,__inference_model_960_layer_call_fn_60989182?
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
#__inference__wrapped_model_60988272?
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
input_11??????????
?2?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60989933
I__inference_autoencoder_layer_call_and_return_conditional_losses_60989849
I__inference_autoencoder_layer_call_and_return_conditional_losses_60990039
I__inference_autoencoder_layer_call_and_return_conditional_losses_60990123
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988418
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988405?
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
.__inference_autoencoder_layer_call_fn_60989944
.__inference_autoencoder_layer_call_fn_60990134
.__inference_autoencoder_layer_call_fn_60990145
.__inference_autoencoder_layer_call_fn_60989955
.__inference_autoencoder_layer_call_fn_60988467
.__inference_autoencoder_layer_call_fn_60988443?
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
K__inference_discriminator_layer_call_and_return_conditional_losses_60988754
K__inference_discriminator_layer_call_and_return_conditional_losses_60990301
K__inference_discriminator_layer_call_and_return_conditional_losses_60988778
K__inference_discriminator_layer_call_and_return_conditional_losses_60990450?
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
0__inference_discriminator_layer_call_fn_60990467
0__inference_discriminator_layer_call_fn_60990484
0__inference_discriminator_layer_call_fn_60988820
0__inference_discriminator_layer_call_fn_60988861?
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
&__inference_signature_wrapper_60989254input_11"?
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
G__inference_dense_412_layer_call_and_return_conditional_losses_60990515?
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
,__inference_dense_412_layer_call_fn_60990522?
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
G__inference_dense_413_layer_call_and_return_conditional_losses_60990553?
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
,__inference_dense_413_layer_call_fn_60990560?
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
G__inference_dense_414_layer_call_and_return_conditional_losses_60990590?
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
,__inference_dense_414_layer_call_fn_60990597?
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
E__inference_dropout_layer_call_and_return_conditional_losses_60990609
E__inference_dropout_layer_call_and_return_conditional_losses_60990614?
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
*__inference_dropout_layer_call_fn_60990619
*__inference_dropout_layer_call_fn_60990624?
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
G__inference_dense_415_layer_call_and_return_conditional_losses_60990655?
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
,__inference_dense_415_layer_call_fn_60990662?
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
G__inference_dense_416_layer_call_and_return_conditional_losses_60990693?
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
,__inference_dense_416_layer_call_fn_60990700?
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
G__inference_dense_417_layer_call_and_return_conditional_losses_60990731?
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
,__inference_dense_417_layer_call_fn_60990738?
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
G__inference_dense_418_layer_call_and_return_conditional_losses_60990769?
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
,__inference_dense_418_layer_call_fn_60990776?
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
G__inference_dense_419_layer_call_and_return_conditional_losses_60990807?
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
,__inference_dense_419_layer_call_fn_60990814?
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
H__inference_flatten_51_layer_call_and_return_conditional_losses_60990820?
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
-__inference_flatten_51_layer_call_fn_60990825?
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
G__inference_dense_420_layer_call_and_return_conditional_losses_60990836?
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
,__inference_dense_420_layer_call_fn_60990843?
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
#__inference__wrapped_model_60988272?%&'()*+,-./01234566?3
,?)
'?$
input_11??????????
? "=?:
8
discriminator'?$
discriminator??????????
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988405t%&'()*>?;
4?1
'?$
input_11??????????
p

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60988418t%&'()*>?;
4?1
'?$
input_11??????????
p 

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60989849r%&'()*<?9
2?/
%?"
inputs??????????
p

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60989933r%&'()*<?9
2?/
%?"
inputs??????????
p 

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60990039y%&'()*C?@
9?6
,?)
'?$
inputs/0??????????
p

 
? "*?'
 ?
0??????????
? ?
I__inference_autoencoder_layer_call_and_return_conditional_losses_60990123y%&'()*C?@
9?6
,?)
'?$
inputs/0??????????
p 

 
? "*?'
 ?
0??????????
? ?
.__inference_autoencoder_layer_call_fn_60988443g%&'()*>?;
4?1
'?$
input_11??????????
p

 
? "????????????
.__inference_autoencoder_layer_call_fn_60988467g%&'()*>?;
4?1
'?$
input_11??????????
p 

 
? "????????????
.__inference_autoencoder_layer_call_fn_60989944e%&'()*<?9
2?/
%?"
inputs??????????
p

 
? "????????????
.__inference_autoencoder_layer_call_fn_60989955e%&'()*<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
.__inference_autoencoder_layer_call_fn_60990134l%&'()*C?@
9?6
,?)
'?$
inputs/0??????????
p

 
? "????????????
.__inference_autoencoder_layer_call_fn_60990145l%&'()*C?@
9?6
,?)
'?$
inputs/0??????????
p 

 
? "????????????
G__inference_dense_412_layer_call_and_return_conditional_losses_60990515e%&4?1
*?'
%?"
inputs??????????
? ")?&
?
0????????? 
? ?
,__inference_dense_412_layer_call_fn_60990522X%&4?1
*?'
%?"
inputs??????????
? "?????????? ?
G__inference_dense_413_layer_call_and_return_conditional_losses_60990553e'(3?0
)?&
$?!
inputs????????? 
? "*?'
 ?
0??????????
? ?
,__inference_dense_413_layer_call_fn_60990560X'(3?0
)?&
$?!
inputs????????? 
? "????????????
G__inference_dense_414_layer_call_and_return_conditional_losses_60990590f)*4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_414_layer_call_fn_60990597Y)*4?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_415_layer_call_and_return_conditional_losses_60990655f+,4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_415_layer_call_fn_60990662Y+,4?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_416_layer_call_and_return_conditional_losses_60990693f-.4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_416_layer_call_fn_60990700Y-.4?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_417_layer_call_and_return_conditional_losses_60990731f/04?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_417_layer_call_fn_60990738Y/04?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_418_layer_call_and_return_conditional_losses_60990769f124?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_dense_418_layer_call_fn_60990776Y124?1
*?'
%?"
inputs??????????
? "????????????
G__inference_dense_419_layer_call_and_return_conditional_losses_60990807e344?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
,__inference_dense_419_layer_call_fn_60990814X344?1
*?'
%?"
inputs??????????
? "???????????
G__inference_dense_420_layer_call_and_return_conditional_losses_60990836\56/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_420_layer_call_fn_60990843O56/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_discriminator_layer_call_and_return_conditional_losses_60988754u+,-./0123456>?;
4?1
'?$
input_12??????????
p

 
? "%?"
?
0?????????
? ?
K__inference_discriminator_layer_call_and_return_conditional_losses_60988778u+,-./0123456>?;
4?1
'?$
input_12??????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_discriminator_layer_call_and_return_conditional_losses_60990301s+,-./0123456<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????
? ?
K__inference_discriminator_layer_call_and_return_conditional_losses_60990450s+,-./0123456<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
0__inference_discriminator_layer_call_fn_60988820h+,-./0123456>?;
4?1
'?$
input_12??????????
p

 
? "???????????
0__inference_discriminator_layer_call_fn_60988861h+,-./0123456>?;
4?1
'?$
input_12??????????
p 

 
? "???????????
0__inference_discriminator_layer_call_fn_60990467f+,-./0123456<?9
2?/
%?"
inputs??????????
p

 
? "???????????
0__inference_discriminator_layer_call_fn_60990484f+,-./0123456<?9
2?/
%?"
inputs??????????
p 

 
? "???????????
E__inference_dropout_layer_call_and_return_conditional_losses_60990609f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
E__inference_dropout_layer_call_and_return_conditional_losses_60990614f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
*__inference_dropout_layer_call_fn_60990619Y8?5
.?+
%?"
inputs??????????
p
? "????????????
*__inference_dropout_layer_call_fn_60990624Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
H__inference_flatten_51_layer_call_and_return_conditional_losses_60990820\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? ?
-__inference_flatten_51_layer_call_fn_60990825O3?0
)?&
$?!
inputs?????????
? "???????????
G__inference_model_960_layer_call_and_return_conditional_losses_60989110{%&'()*+,-./0123456>?;
4?1
'?$
input_11??????????
p

 
? "%?"
?
0?????????
? ?
G__inference_model_960_layer_call_and_return_conditional_losses_60989134{%&'()*+,-./0123456>?;
4?1
'?$
input_11??????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_960_layer_call_and_return_conditional_losses_60989490y%&'()*+,-./0123456<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????
? ?
G__inference_model_960_layer_call_and_return_conditional_losses_60989719y%&'()*+,-./0123456<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_model_960_layer_call_fn_60989182n%&'()*+,-./0123456>?;
4?1
'?$
input_11??????????
p

 
? "???????????
,__inference_model_960_layer_call_fn_60989229n%&'()*+,-./0123456>?;
4?1
'?$
input_11??????????
p 

 
? "???????????
,__inference_model_960_layer_call_fn_60989742l%&'()*+,-./0123456<?9
2?/
%?"
inputs??????????
p

 
? "???????????
,__inference_model_960_layer_call_fn_60989765l%&'()*+,-./0123456<?9
2?/
%?"
inputs??????????
p 

 
? "???????????
&__inference_signature_wrapper_60989254?%&'()*+,-./0123456B??
? 
8?5
3
input_11'?$
input_11??????????"=?:
8
discriminator'?$
discriminator?????????