%% Using the UQLab client together with UM-Bridge
% UQLab is a general purpose Uncertainty Quantification framework based on 
% Matlab. It is made of open-source scientific modules which are smoothly
% connected to perform uncertainty quantification through Monte Carlo 
% simulation, sensitivity analysis, reliability analysis (computation of 
% rare event probabilities), surrogate models (polynomial chaos expansions, 
% Kriging, etc.), Bayesian inversion/calibration and many more.

% UQLab fully supports UM-Bridge models. After registering at 
% `https://www.uqlab.com/`, downloading and installing UQLab as indicated 
% on the UQLab website, you can start UQLab using
uqlab

% To run an UM-Bridge model, install Docker Desktop and start a docker 
% model, e.g., the Euler-Bernoulli beam model 
% (https://um-bridge-benchmarks.readthedocs.io/en/docs/forward-benchmarks/muq-beam-propagation.html), 
% using the terminal command
% >> docker run -it -p 4243:4243 linusseelinger/benchmark-muq-beam-propagation:latest

% Add UM-Bridge to your Matlab path and create an UM-Bridge model as in the 
% section "Matlab client" of
% https://um-bridge-benchmarks.readthedocs.io/en/docs/umbridge/clients.html#matlab-client:
addpath(fullfile('\YOUR\FULL\PATH\TO\UMBRIDGE\INSTALLATION', 'matlab')) % modify with the correct path
model = HTTPModel('http://localhost:4243', 'forward');

% The following lines create the corresponding UQLab model object:
ModelOpts.mHandle = @(X) model.evaluate(X);
EulerBernoulliBeamModel = uq_createModel(ModelOpts);

% Additional parameters can also be passed to the model, see the UQLab User 
% Manual: The Model Module, Section 2.1.4
% (https://uqftp.ethz.ch/uqlab_doc_pdf/2.1.0/UserManual_Model.pdf).

% The model can now be evaluated for one or more sets of input parameters. 
% For example, the three-dimensional Euler-Bernoulli beam model with 
% parameter space [1.00 ,1.05]^3 can be evaluated at two parameter sets as 
% follows:
X = [1.00, 1.00, 1.01; ...
     1.04, 1.01, 1.05];
Y = uq_evalModel(EulerBernoulliBeamModel, X);

% Each row of `X` and `Y` correspond to one set of input parameters.
% In general, let `N` denote the number of sample sets, and let `N_in` and 
% `N_out` denote the input and output dimension of the model, respectively.
% Then `X` has dimension `N x N_in`, and `Y` has dimension `N x N_out`.

% The UQLab model object can be used further for all types of UQ analyses, 
% as explained in detail in the UQLab documentation 
% (https://www.uqlab.com/documentation).
