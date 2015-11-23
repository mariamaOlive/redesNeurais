function [Y,Xf,Af,E,perf]=privatesim(net,varargin)
%SIM Simulate a neural network.
%
%  SIM(NET,X) takes a network NET and inputs X and returns the outputs
%  Y generated by the network.  This syntax is equivalent to NET(X).
%
%  [Y,Xf,Af] = <a href="matlab:doc sim">sim</a>(NET,X,Xi,Ai) takes a dynamic network NET, inputs X,
%  and initial input and layer delays Xi and Ai.  It returns the outputs Y
%  and final input and layer delays states Xf and Af.
%
%  <a href="matlab:doc sim">sim</a> arguments can have two formats: matrices, for static
%  problems and networks with single inputs and outputs, and cell arrays
%  for multiple timesteps and networks with multiple inputs and outputs.
%
%  The matrix format is as follows:
%    X  - RxQ matrix
%    Y  - UxQ matrix.
%  Where:
%    Q  = number of samples
%    R  = number of elements in the network's input
%    U  = number of elements in the network's output
%
%  The cell array format is most general:
%    X  - NixTS cell array, each element X{i,ts} is an RixQ matrix.
%    Xi - NixID cell array, each element Xi{i,k} is an RixQ matrix.
%    Ai - NlxLD cell array, each element Ai{i,k} is an SixQ matrix.
%    Y  - NOxTS cell array, each element Y{i,ts} is a UixQ matrix.
%    Xf - NixID cell array, each element Xf{i,k} is an RixQ matrix.
%    Af - NlxLD cell array, each element Af{i,k} is an SixQ matrix.
%  Where:
%    TS = number of time steps
%    Ni = NET.<a href="matlab:doc nnproperty.net_numInputs">numInputs</a>
%    Nl = NET.<a href="matlab:doc nnproperty.net_numLayers">numLayers</a>, 
%    No = NET.<a href="matlab:doc nnproperty.net_numOutputs">numOutputs</a>
%    ID = NET.<a href="matlab:doc nnproperty.net_numInputDelays">numInputDelays</a>
%    LD = NET.<a href="matlab:doc nnproperty.net_numLayerDelays">numLayerDelays</a>
%    Ri = NET.<a href="matlab:doc nnproperty.net_inputs">inputs</a>{i}.<a href="matlab:doc nnproperty.input_size">size</a>
%    Si = NET.<a href="matlab:doc nnproperty.net_layers">layers</a>{i}.<a href="matlab:doc nnproperty.layer_size">size</a>
%    Ui = NET.<a href="matlab:doc nnproperty.net_outputs">outputs</a>{i}.<a href="matlab:doc nnproperty.output_size">size</a>
%
%  The columns of Xi, Xf, Ai, and Af are ordered from oldest delay
%  condition to most recent:
%    Xi{i,k} = input i at time ts=k-ID.
%    Xf{i,k} = input i at time ts=TS+k-ID.
%    Ai{i,k} = layer output i at time ts=k-LD.
%    Af{i,k} = layer output i at time ts=TS+k-LD.
%
%  [Y,Pf,Af] = SIM(net,{Q TS},Pi,Ai) is used for networks
%  which do not have an input, such as Hopfield networks
%  when cell array notation is used.
%
%  Here a static feedforward network is created, trained on some data, then
%  simulated using SIM and network notation.
%
%    [x,t] = <a href="matlab:doc simplefit_dataset">simplefit_dataset</a>;
%    net = <a href="matlab:doc feedforwardnet">feedforwardnet</a>(10);
%    net = <a href="matlab:doc train">train</a>(net,x,t);
%    y1 = <a href="matlab:doc sim">sim</a>(net,x)
%    y2 = net(x)
%
%  Here a dynamic NARX network is created, trained, and simulated on
%  time series data.
%
%   [X,T] = <a href="matlab:doc simplenarx_dataset">simplenarx_dataset</a>;
%   net = <a href="matlab:doc narxnet">narxnet</a>(1:2,1:2,10);
%   <a href="matlab:doc view">view</a>(net)
%   [Xs,Xi,Ai,Ts] = <a href="matlab:doc preparets">preparets</a>(net,X,{},T);
%   net = <a href="matlab:doc train">train</a>(net,Xs,Ts,Xi,Ai);
%   Y1 = <a href="matlab:doc sim">sim</a>(net,Xs,Xi,Ai)
%   Y2 = net(Xs,Xi,Ai)
%
%  <strong>Simulation with Parallel Computing</strong>
%
%  Parallel Computing Toolbox allows Neural Network Toolbox to simulate
%  networks faster and on larger datasets than can fit on one PC.
%
%  Here simulation automatically happens across MATLAB parallel workers.
%
%    parpool
%    [X,T] = vinyl_dataset;
%    net = feedforwardnet(140,'trainscg');
%    net = train(net,X,T,'UseParallel','yes');
%    Y = net(X,'UseParallel','yes');
%
%  Use Composite values to distribute the data manually, and get back
%  the results as a Composite value.  If the data is loaded as it is
%  distributed then while each peice of the dataset must fit in RAM, the
%  entire dataset is only limited by the number of workers RAM.
%
%    Xc = Composite;
%    for i=1:numel(Xc)
%      Xc{i} = X + rand(size(X))*0.1; % (Use real data instead of random)
%    end
%    Yc = net(Xc);
%    Y = cat(2,Yc{:});
%
%  Networks can be simulated using the current GPU device, if it is
%  supported by the Parallel Computing Toolbox.  This is efficient for
%  large static problems or dynamic problems with many series.
%
%    Y = net(X,'UseGPU','yes');
%
%  To put the data on a GPU manually, and get the results on the GPU,
%  the network must be static and have a single input and output:
%
%    Xgpu = gpuArray(X);
%    Ygpu = net(Xgpu);
%    Y = gather(Ygpu);
%
%  To run in parallel, with workers associated with unique GPUs taking
%  advantage of that hardware, while the rest of the workers use CPUs:
%
%    Y = net(X,'UseParallel','yes','UseGPU','yes');
%
%  Only using workers with unique GPUs may result in higher speed, as CPU
%  workers may not keep up.
%
%    Y = net(X,'UseParallel','yes','UseGPU','only');
%
%  Use the 'ShowResources' option to verify the computing resources used.
%
%    y = net(...,'ShowResources','yes');
%
%  See also INIT, REVERT, ADAPT, TRAIN

% Copyright 1992-2014 The MathWorks, Inc.

% CHECK AND FORMAT ARGUMENTS
% --------------------------

% Network
if nargin < 1, error(message('nnet:Args:NotEnough')); end
if ~isa(net,'network')
  error('nnet:train:arguments','First argument is not a neural network.');
end
net = struct(net);
net.trainFcn = ''; % Disable training related setup
[~,zeroDelayLoop] = nn.layer_order(net);
 if zeroDelayLoop, error(message('nnet:NNet:ZeroDelayLoop')); end
 
 % PICK CALCULATION MODE

% Undocumented Testing API.
% This API may be altered or removed at any time.
% Specify calculation mode by name as last argument of train.
if ~isempty(varargin) && isstruct(varargin{end}) && isfield(varargin{end},'name')
  calcMode = nncalc.defaultMode(net,varargin{end}); varargin(end) = [];
  calcMode.options = nnet.options.calc.defaults;
  calcMode.options.showResources = 'yes';

% Documented API
% Recommended API for customers and most testing.
% Use optional parameter/value pairs to pick calculation mode.  
else
  [varargin,nameValuePairs] = nnet.arguments.extractNameValuePairs(varargin);
  [calcMode,err] = nncalc.options2Mode(net,nameValuePairs);
  if ~isempty(err), error('nnet:train:arguments',err); end
end
problem = calcMode.netCheck(net,calcMode.hints,false,false);
if ~isempty(problem), error(problem); end

% Check Composite Data for consistency
nargs = numel(varargin);
if nargs >= 1
  isComposite = isa(varargin{1},'Composite');
else
  isComposite = false;
end
for i=2:nargs
  if isComposite ~= isa(varargin{i},'Composite')
    error('nnet:sim:Composite','Data values must be all Composite or not.');
  end
end

% Check gpuArray data for consistency
if nargs >= 1
  isGPUArray = isa(varargin{1},'gpuArray');
else
  isGPUArray = false;
end
for i=2:nargs
  vi = varargin{i};
  if ~isempty(vi) && (isGPUArray ~= isa(vi,'gpuArray')) 
    error('nnet:sim:Composite','Data values must be all gpuArray or not.');
  end
end

% Fill in missing data consistent with type
if isComposite
  emptyCell = Composite;
  oneCell = Composite;
  for i=1:numel(emptyCell)
    emptyCell{i} = {};
    oneCell{i} = {1};
  end
else
  emptyCell = {};
  oneCell = {1};
end
if (nargs < 1), X = emptyCell; else X = varargin{1}; end
if (nargs < 2), Xi = emptyCell; else Xi = varargin{2}; end
if (nargs < 3), Ai = emptyCell; else Ai = varargin{3}; end
if (nargs < 4), T = emptyCell; else T = varargin{4}; end
if (nargs < 5) || isempty(varargin{5}), EW=oneCell; else EW=varargin{5}; end
if isComposite
  for i=1:numel(X)
    if ~exist(X,i), X{i} = {}; end
    if ~exist(T,i), T{i} = {}; end
    if ~exist(Xi,i), Xi{i} = {}; end
    if ~exist(Ai,i), Ai{i} = {}; end
    if ~exist(EW,i), EW{i} = {1}; end
  end
end
% X a Matrix or Cell? Use to format Y later.
if isComposite
  spmd, xMatrix = ~iscell(X) && ~isa(X,'gpuArray'); end
elseif isGPUArray
  xMatrix = false;
else
  xMatrix = ~iscell(X);
end

% Convert explicit timesteps to inputs
if ~isComposite && ~isGPUArray
  xMatrix = ~iscell(X);
  if (net.numInputs == 0)
    if xMatrix && isscalar(X)
      % Q
      X = {zeros(0,X)};
    elseif ~xMatrix && isscalar(X) && isscalar(X{1})
      % {TS}
      X = cell(1,X{1});
    elseif xMatrix && (ndims(X)==2) && all(size(X)==[1 2])
      % [Q TS]
      Q = X(1);
      TS = X(2);
      X = cell(1,TS);
      for i=1:TS,X{i} = zeros(1,Q); end
      xMatrix = false;
    elseif ~xMatrix && (ndims(X)==2) && all(size(X)==[1 2]) ...
        && isscalar(X{1}) && isscalar(X{2})
      % {Q TS}
      Q = X{1}; TS = X{2};
      X = cell(1,TS);
      for i=1:TS,X{i} = zeros(1,Q); end
      xMatrix = false;
    end
  end
  X = nntype.data('format',X,'Inputs');
end

% NNET 5.1 Backward Compatibility
if ~isComposite
  tMatrix = ~iscell(T);
  if ~isempty(T), T = nntype.data('format',T,'Targets'); end
end

% SIMULATE NETWORK
% ----------------

% Format Data Arguments
if isComposite
  spmd
    [data,err] = simData(net,X,Xi,Ai,T,EW);
    if ~isempty(regexp(err,'^nnet:','once')), error(message(err)), end
    if ~isempty(err), nnerr.throw(err), end
  end
else
  [data,err] = simData(net,X,Xi,Ai,T,EW);
  if ~isempty(regexp(err,'^nnet:','once')), error(message(err)), end
  if ~isempty(err), nnerr.throw(err), end
end

% Setup Calculation Mode
[calcLib,calcNet,net,resourceText] = nncalc_setup(calcMode,net,data);
if ~isempty(resourceText)
  disp(' ')
  disp('Computing Resources:')
  nntext.disp(resourceText)
  disp(' ')
end
isParallel = isa(calcLib,'Composite');
doAf = (nargout == 3);
doXf = (nargout >= 2);

% Simulate Parallel
if isParallel
  spmd
    [Y,Af] = simPerWorker(calcLib,calcNet,doAf);
    if isComposite
      if doXf, Xf = getXf(net,data); end
      if (xMatrix)
        if isempty(Y), Y = zeros(sum(nn.output_sizes(net)),data.Q); else Y = cell2mat(Y); end
      end
    end
    if (labindex == 1), mainWorkerInd = calcLib.mainWorkerInd; end
  end
  if ~isComposite
    mainWorkerInd = mainWorkerInd{1};
    Y = Y{mainWorkerInd};
    if xMatrix
      if isempty(Y), Y = zeros(sum(nn.output_sizes(net)),data.Q); else Y = cell2mat(Y); end
    end
    if doXf, Xf = getXf(net,data); end
    if doAf, Af = Af{mainWorkerInd}; end
  end
  
% Simulate Single Process
else
  [Y,Af] = simPerWorker(calcLib,calcNet,doAf);
  if doXf, Xf = getXf(net,data); end
  if (xMatrix)
    if isempty(Y), Y = zeros(sum(nn.output_sizes(net)),data.Q); else Y = cell2mat(Y); end
  end
end

% NNET 5.1 Backward Compatibility
if ~isComposite
  if nargout >= 4
    E = gsubtract(data.T,Y);
    if (nargout>4) && (tMatrix)
      if (data.TS==0) || (net.numOutputs == 0)
        E = zeros(sum(nn.output_sizes),Q);
      else
        E = cell2mat(E);
      end
    end
  end
  if nargout >= 5
    perf = feval(net.performFcn,net,data.T,Y,data.EW,net.performParam);
  end
end

%====================================================================

function [Y,Af] = simPerWorker(calcLib,calcNet,doAf)

ws = warning('off','parallel:gpu:kernel:NullPointer');
try
  if doAf
    [Y,Af] = calcLib.y(calcNet);
  else
    Y = calcLib.y(calcNet);
    Af = [];
  end
catch me
  warning(ws); rethrow(me); % Ensure warning state is reverted
end
warning(ws);

%====================================================================

function [data,err] = simData(net,X,Xi,Ai,T,EW)

data = struct;
err = '';

if ~isa(X,'gpuArray')
  [err,net,X,Xi,Ai,T,EW,Q,TS] = simDataCellOfMatrix(net,X,Xi,Ai,T,EW);
  if ~isempty(err), return, end
  data.format = 'CELLofMATRIX';
else
  [err] = simDataCellOfGPUArray(net,X,Xi,Ai,T,EW);
  if isempty(err)
    [err,net,X,Xi,Ai,T,EW,Q,TS] = simDataCellOfGPUArray(net,X,Xi,Ai,T,EW);
    data.format = 'CELLofGPU';
    
  else
    [err2,net,X,Xi,Ai,T,EW,Q,TS] = simDataGPUArray(net,X,Xi,Ai,T,EW);
    if ~isempty(err2), return, end
    err = '';
    data.format = 'NNDATA2GPU';
  end
end

data.X = X;
data.Xi = Xi;
data.Ai = Ai;
data.Q = Q;
data.TS = TS;

% NNET 5.1 Compatibility
data.T = T;
data.EW = EW;


%====================================================================

function [err,net,X,Xi,Ai,T,EW,Q,TS] = simDataCellOfMatrix(net,X,Xi,Ai,T,EW)

err = '';

if ~isempty(X), X = nntype.data('format',X,'Inputs'); end
if ~isempty(Xi), Xi = nntype.data('format',Xi,'Input delay states'); end
if ~isempty(Ai), Ai = nntype.data('format',Ai,'Layer delay states'); end

% Q
if ~isempty(X) && (size(X{1},2) > 0)
  Q = size(X{1},2);
elseif ~isempty(Xi) && ~isempty(Xi{1})
  Q = size(Xi{1},2);
elseif ~isempty(Ai) && ~isempty(Ai{1})
  Q = size(Ai{1},2);
else
  Q = 0;
end

% TS
if (net.numInputs == 0) && (size(X,1) == 1) && (size(X{1},1)==0)
  TS = size(X,2);
  X = cell(0,TS);
elseif size(X,2) > 0
  TS = size(X,2);
else
  TS = 0;
end

% Input
if isempty(X) || (net.numInputs == 0)
  X = cell(net.numInputs,TS);
  for i=1:net.numInputs
    for ts=1:TS
      X{i,ts} = zeros(net.inputs{i}.size,Q);
    end
  end
end
err = nntype.data('check',X);
if ~isempty(err), err = ['Inputs: ' err]; return; end
[Xn,Xq,Xts,Xs] = nnfast.nnsize(X);
if isempty(X), Xq = Q; end
if (Xs == 1) && (net.numInputs ~= 1)
  Nn = zeros(1,net.numInputs);
  for i=1:net.numInputs,Nn(i) = net.inputs{i}.size; end
  if (Xn == sum(Nn))
    X2 = cell(net.numInputs,Xts);
    for ts=1:Xts
      X2(:,ts) = mat2cell(X{1,ts},Nn,Xq);
    end
    X = X2;
    Xn = Nn;
    Xs = net.numInputs;
  end
end
if (Xs ~= net.numInputs)
  err = 'Number of inputs does not match net.numInputs.'; return;
end

% Target
if isempty(T)
  targetIndices = find(net.outputConnect);
  T = cell(net.numOutputs,Xts);
  for i=1:net.numOutputs
    ii = targetIndices(i);
    ti = NaN(net.outputs{ii}.size,Xq);
    for j=1:Xts, T{i,j} = ti; end
  end
end
err = nntype.data('check',T);
if ~isempty(err), err = ['Targets: ' err]; return; end
[Tn,Tq,Tts,Ts] = nnfast.nnsize(T);
if ((Ts == 0) || (Tts ==0)) && (Tq == 0)
  Tq = Xq;
end
if (Tq ~= Xq)
  err = 'Inputs and targets have different numbers of samples.'; return
end
if (Tts ~= Xts)
  err = 'Inputs and targets have different numbers of timesteps.'; return
end
if (Ts == 1) && (net.numOutputs ~= 1)
  Nn = zeros(1,net.numOutputs);
  outputInd = find(net.outputConnect);
  for i=1:net.numOutputs,Nn(i) = net.outputs{outputInd(i)}.size; end
  if (Tn == sum(Nn))
    T2 = cell(net.numOutputs,Xts);
    for ts=1:Xts
      T2(:,ts) = mat2cell(T{1,ts},Nn,Tq);
    end
    T = T2;
    Tn = Nn;
    Ts = net.numOutputs;
  end
end
if (Ts ~= net.numOutputs)
  err = 'Number of targets does not match net.numOutputs.'; return;
end

% Input States
if isempty(Xi)
  Xi = cell(net.numInputs,net.numInputDelays);
  for i=1:net.numInputs
    xi = zeros(net.inputs{i}.size,Xq);
    for j=1:net.numInputDelays, Xi{i,j} = xi; end
  end
end
err = nntype.data('check',Xi);
if ~isempty(err), err = ['Input states: ' err]; return; end
[Xin,Xiq,Xits,Xis] = nnfast.nnsize(Xi);
if ((Xis == 0) || (Xits ==0)) && (Xiq == 0)
  Xiq = Xq;
end
if (Xiq ~= Xq)
  err = 'Inputs and input states have different numbers of samples.'; return
end
if (Xis ~= net.numInputs)
  err = 'Number of input states does not match net.numInputs.'; return;
end
if (Xits ~= net.numInputDelays)
  err = 'Number of input state timesteps does not match net.numInputDelays.'; return
end
if (Xis > 0) && (Xits > 0)
  for i=1:Xis
    if Xin(i) ~= net.inputs{i}.size
      err = 'Input state sizes does not match net.inputs{:}.size.'; return;
    end
  end
end

% Layer States
if isempty(Ai)
  Ai = cell(net.numLayers,net.numLayerDelays);
  for i=1:net.numLayers
    ai = zeros(net.layers{i}.size,Xq);
    for j=1:net.numLayerDelays, Ai{i,j} = ai; end
  end
end
err = nntype.data('check',Ai);
if ~isempty(err), err = ['Layer states: ' err]; return; end
[Ain,Aiq,Aits,Ais] = nnfast.nnsize(Ai);
if ((Ais == 0) || (Aits ==0)) && (Aiq == 0)
  Aiq = Xq;
end
if (Aiq ~= Xq)
  err = 'Inputs and layer states have different numbers of samples.'; return
end
if (Ais ~= net.numLayers)
  err = 'Number of layers states does not match net.numLayers.'; return;
end
if (Aits ~= net.numLayerDelays)
  err = 'Number of layer state timesteps does not match net.numLayerDelays.'; return
end

%====================================================================

function [err,net,X,Xi,Ai,T,EW,Q,TS] = simDataCellOfGPUArray(net,X,Xi,Ai,T,EW)

err = '';
if (net.numInputDelays + net.numLayerDelays) > 0
  err = 'nnet:parallel:gpuArrayDataDynamicNetwork';
  return
end
if (net.numInputs ~= 1) || (net.numOutputs ~= 1)
  err = 'nnet:parallel:gpuArrayDataMultipleIONetwork';
  return
end

% Q
if ~isempty(X)
  Q = size(X,2);
else
  Q = 0;
end

% TS
TS = 1;

% Network dimensions
Ni = net.inputs{1}.size;

% Expand empty values
if isempty(X), X = {gpuArray(nan(Ni,Q))}; end
if isempty(Xi), Xi = cell(1,0); end
if isempty(Ai), Ai = cell(net.numLayers,0); end
if ~iscell(X), X = {X}; end

% Check X
if any(size(X) ~= [1 TS])
  err = 'Incorrectly sized inputs X.';
  return
end
for i=1:numel(X)
  x = X{i};
  if size(x,1) ~= Ni
    if (Ni == 0)
      err = 'Network must be configured with CONFIGURE before training with gpuArray data.';
    else
      err = 'Incorrectly sized inputs X.';
      return
    end
  end
  if size(x,2) ~= Q
    err = 'Incorrectly sized inputs X.';
    return
  end
end

% Check Xi
if ~isempty(Xi)
  err = 'Incorrectly sized input states Xi.';
  return
end

% Check Ai
if ~isempty(Ai)
  err = 'Incorrectly sized layer states Ai.';
  return
end

T = {};
EW = {};

%====================================================================

function [err,net,X,Xi,Ai,T,EW,Q,TS] = simDataGPUArray(net,X,Xi,Ai,T,EW)

err = '';

% Infer Precision
if ~isempty(X)
  precision = class(gather(X(1)));
elseif ~isempty(Xi)
  precision = class(gather(Xi(1)));
elseif ~isempty(Ai)
  precision = class(gather(Ai(1)));
else
  precision = 'double';
end

% QQ
QQs = [size(X,1) size(Xi,1) size(Ai,1)];
QQs(QQs == 0) = [];
QQ = max([0 QQs]);
if any(QQs ~= QQ)
  err = 'Number of samples (rows of gpuArrays) of data arguments do not match.';
  return
end

% Q
if ~isempty(X)
  Qv = X;
elseif ~isempty(Xi)
  Qv = Xi;
elseif ~isempty(Ai)
  Qv = Ai;
else
  Qv = [];
end
realRows = gather(any(isfinite(Qv),2));
Q = find(realRows,1,'last');

% Network dimensions
Ni = sum(nn.input_sizes(net));
Nl = sum(nn.layer_sizes(net));
NID = net.numInputDelays;
NLD = net.numLayerDelays;
anyInputsZero = any(nn.input_sizes(net)==0);

% Infer TS
Ni_TS = size(X,2);
if (Ni_TS == 0)
  TS = 0;
elseif (Ni > 0)
  TS = Ni_TS / Ni;
  if (TS ~= floor(TS))
    if anyInputsZero
      err = 'Input data size  (gpuArray columns) does not match input sizes. Fix data or CONFIGURE network.';
    else
      err = 'Input data size  (gpuArray columns) does not match input sizes.';
    end
    return;
  end
else
  TS = 0;
end

% Expand empty values
if isempty(X), X = gpuArray(nan(QQ,Ni*TS,precision)); end
if isempty(Xi), Xi = gpuArray(nan(QQ,Ni*NID,precision)); end
if isempty(Ai), Ai = gpuArray(nan(QQ,Nl*NLD,precision)); end

% Check sizes
if any(size(X) ~= [QQ Ni*TS])
  if anyInputsZero
    err = 'Input data size  (gpuArray columns) does not match input sizes. Fix data or CONFIGURE network.';
  else
    err = 'Input data size  (gpuArray columns) does not match input sizes.';
  end
  return
end
if any(size(Xi) ~= [QQ Ni*NID])
  err = 'Input state size  (gpuArray columns) does not match input sizes times input delay states.';
end
if any(size(Ai) ~= [QQ Nl*NLD])
  err = 'Layer state size  (gpuArray columns) does not match layers sizes times layer delay states.';
end

T = {};
EW = {};

%====================================================================

function Xf = getXf(net,data)

if ~isempty(data)
  Xf = cell(net.numInputs,net.numInputDelays);
  for ts=1:net.numInputDelays
    x_ts = ts+data.TS-net.numInputDelays;
    if (x_ts) < 1
      xi_ts = x_ts + net.numInputDelays;
      Xf(:,ts) = data.Xi(:,xi_ts);
    else
      Xf(:,ts) = data.X(:,x_ts);
    end
  end
else
  Xf = [];
end

%====================================================================

function [calcLib,calcNet,net,resourceText] = nncalc_setup(calcMode,net,data)
% Setup calculation mode, net, data & hints for non-parallel calculations
% Replace with call to nncalc.setup when compiler dependencies are updated

% Copyright 2012-2013 The MathWorks, Inc.

% Setup Step 1: On Main Thread Only
[calcMode,calcNet,calcData,calcHints,net,resourceText] = nncalc.setup1(calcMode,net,data);

% Setup Step 2: On Each Worker, if using Parallel Calculation Mode
if isa(calcMode,'Composite');
  spmd
    % Finish setup for parallel mode
    [calcLib,calcNet] = nncalc.setup2(calcMode,net,calcData,calcHints);
  end
else
  % Finish setup for MATLAB, MEX, GPU and other non-parallel modes
  [calcLib,calcNet] = nncalc.setup2(calcMode,calcNet,calcData,calcHints);
end
