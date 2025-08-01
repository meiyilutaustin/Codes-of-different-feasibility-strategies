

% dcopf_make_coefficients_compact, you can get dcopf_data.mat
%  Builds (Q, c, Aeq, Beq, Aineq, Bineq, bbineq, ABbineq)
%  and solves the resulting QP             –  Compact edition –
%  Requires MATPOWER (≥ v7)
%  min ½ pgᵀ Q pg +c Pg
%  s.t. Aeq@pg+Beq@pd==0;Aineq@pg+Bineq@pd+bbineq<=0

clear;
clc;
%case_ACTIVSg500,2000
%case1354pegase
format   short
mga = loadcase ('case_ACTIVSg200'); 
define_constants;
baseMVA = mga.baseMVA;

%% ------------------------- 1. only keep the generators and branch on service
on_branch = mga.branch(:,BR_STATUS)==1;
on_gen    = mga.gen(:,GEN_STATUS)==1;

mga.branch   = mga.branch(on_branch , :);
mga.gen      = mga.gen(on_gen      , :);
mga.gencost  = mga.gencost(on_gen  , :);   % <-- keep gencost in sync



nb = length (mga.bus(:,1));        %number of bus
nr = length (mga.branch(:,1));     %number of branch
ng = length (mga.gen(:,1));        %number of gen

%% ------------------------- 2.change the name of bus and gen and branch so that they are starting from 1....nb--------%
for i=1:nb
    newname=i;
    oldname=mga.bus(i,1);
    mga.bus(i,1)=newname;
    for g=1:ng
        if mga.gen(g,1)==oldname
            mga.gen(g,1)=newname;
        end
    end
    for r=1:nr 
        if mga.branch(r,1)==oldname
            mga.branch(r,1)=newname;
        end
        if mga.branch(r,2)==oldname
            mga.branch(r,2)=newname;
        end
    end
end



%% ------------------------- 3.relax fixed-output units a bit more
fix = mga.gen(:,PMAX) - mga.gen(:,PMIN) < 1e-3;
mga.gen(fix,PMIN) = 0;




%% ------------------------- 4. set the constant coefficient to zeros(won't affect the optimization)
for i=1:ng
    mga.gencost(i,7)=0;%au^2+bu+c, c=0
end











%% ------------------------- 5. Cost matrices ------------------------- %%
gc      = mga.gencost;
%––– Make sure we really have polynomial costs ––––––––––––––––––––––––%
if any(gc(:, MODEL) ~= 2)
    error('This script expects all generators to use the POLYNOMIAL cost model (MODEL = 2).');
end
if any(gc(:, NCOST) < 2)
    error('Generator costs must be at least linear (NCOST ≥ 2).');
end
a2      = gc(:, COST   );            % quadratic term
a1      = gc(:, COST+1 );            % linear term

Q       = 2*diag(a2);                % ½ pgᵀ Q pg   → factor 2
c       = a1;

%% ------------------------- 6. Equality ------------------------------ %%
ng      = size(mga.gen ,1);
nb      = size(mga.bus ,1);
pd      = mga.bus(:, PD);

Aeq     = ones(1, ng);               % Σ pg
Beq     = -ones(1, nb);              % -Σ pd   so  Aeq pg + Beq pd = 0

%% ------------------------- 7. Inequalities -------------------------- %%
% ----- Line-flow limits with PTDF ---------------------------------------
nl        = size(mga.branch, 1);
slack_bus = find(mga.bus(:, BUS_TYPE) == REF, 1, 'first');
PTDF      = makePTDF(baseMVA, mga.bus, mga.branch, slack_bus);

G   = sparse(mga.gen(:, GEN_BUS), 1:ng, 1, nb, ng);   % bus → gen map
% mga.branch(:, RATE_A) = 500;
Fmax = mga.branch(:, RATE_A);
if max(Fmax) < 50           % heuristic: ratings look like p.u. numbers
    Fmax = Fmax * baseMVA;  % convert to MW
end

% • Zero, Inf or NaN ⇒ “unlimited”, therefore ignore
finite_mask   =  isfinite(Fmax) & (Fmax > 0);
if any(finite_mask)          % at least one binding line limit?
    PTDF_f   = PTDF(finite_mask, :);                 % nk × nb
    Fmax_f   = Fmax(finite_mask);                    % nk × 1

    Aline_f  =  PTDF_f * G;                          % nk × ng
    Bline_f  = -PTDF_f;                              % nk × nb
    bline_f  = -Fmax_f;                              % nk × 1

    
    

    % two-sided |flow| ≤ Fmax ;|PTDFg@pg-PTDFd@pd|≤ Fmax;
    % PTDFg@pg-PTDFd@pd≤ Fmax, -Fmax≤ PTDFg@pg-PTDFd@pd;
    % PTDFg@pg-PTDFd@pd- Fmax ≤0, -PTDFg@pg+PTDFd@pd-Fmax≤ 0;
    % Aline_f@pg+Bline_f@pd+bline_f<=0, -Aline_f@pg-Bline_f@pd+bline_f<=0
    Aineq  = [  Aline_f ;  -Aline_f  ];              % 2·nk × ng
    Bineq  = [  Bline_f ;  -Bline_f  ];              % 2·nk × nb
    bbineq = [  bline_f ;   bline_f  ];              % 2·nk × 1
    flag =1;
else
    % no finite limits → start with empty matrices
    Aineq  = sparse(0, ng);
    Bineq  = sparse(0, nb);
    bbineq = [];
    flag=0;
end

% ---------- Generator limits  pg ≤ Pmax  &  -pg ≤ -Pmin ---------------
Pmin   = mga.gen(:, PMIN);
Pmax   = mga.gen(:, PMAX);

Aineq  = [ Aineq ;   eye(ng) ; -eye(ng) ];
Bineq  = [ Bineq ;  zeros(2*ng, nb)     ];
bbineq = [ bbineq; -Pmax ;  Pmin        ];



%% ------------------------- 8. Solve QP ------------------------------ %%

Aqp = Aineq;                         % (can stay sparse)
bqp = full(-Bineq*pd - bbineq);      % make *dense*
beq = full(-Beq*pd);



% %% -------- DEBUG: does MATPOWER’s optimum satisfy our rows? ----------
% dcopf_sol  = dcopf(mga);                     % MATPOWER’s own model
% Pg_mp      = dcopf_sol.gen(:, PG);           % optimal dispatch in MW
% residual   = Aqp*Pg_mp -bqp
% % residual   = Aineq*Pg_mp + Bineq*pd + bbineq;
% [violate, row] = sort(residual, 'descend');  % >0  ⇒ row is violated
% 
% if violate(1) <= 1e-6
%     disp('All compact rows satisfied by MATPOWER solution → model is fine.');
% else
%     fprintf('\n!!!  %d rows violated by MATPOWER solution  !!!\n', ...
%             nnz(residual > 1e-6));
%     fprintf('   row        max-lhs   (should be ≤ 0)\n');
%     for k = 1:min(10, numel(row))
%         fprintf('%6d   %+11.4f\n', row(k), violate(k));
%     end
% end
% return



opts = optimoptions('quadprog','Display','off');
[pg_opt, fval, exitflag] = quadprog(Q, c, Aqp, bqp, Aeq, beq, [], [], [], opts);
if exitflag~=1, warning('quadprog did not converge, flag %d', exitflag); end

%% ------------------------- 9. Compare with MATPOWER dcopf solution ------------------------------ %%
dcopf_result = dcopf(mga);
optimal_gen_matpower = dcopf_result.gen(:, PG);     % ng × 1 vector  (PG = 2)

% ---------- 9a. Generator-output comparison ----------
fprintf('\nComparison of generator outputs (MW):\n');
fprintf(' Gen  QP-soln    MATPOWER    Diff\n');
fprintf('----  ---------- ----------  --------\n');
for k = 1:ng
    fprintf('%3d  %10.3f %10.3f  %+8.4f\n', ...
            k, pg_opt(k), optimal_gen_matpower(k), ...
            pg_opt(k)-optimal_gen_matpower(k));
end
fprintf('----  ---------- ----------  --------\n');
fprintf('Max |diff| = %.6f MW\n', max(abs(pg_opt - optimal_gen_matpower)));
fprintf('ℓ₂  norm  = %.6f MW\n\n', norm(pg_opt - optimal_gen_matpower));

% ---------- 9b. Objective-value (cost) comparison ----------
obj_qp = 0.5 * pg_opt.'            * Q * pg_opt + c.' * pg_opt;      % $/h
obj_mp = dcopf_result.f;                                             % $/h  (MATPOWER returns this)

fprintf('Objective-value comparison ($/h):\n');
fprintf('  QP solver cost      : %12.2f\n', obj_qp);
fprintf('  MATPOWER DCOPF cost : %12.2f\n', obj_mp);
fprintf('  Difference          : %+12.6f  (%.4g%%)\n', ...
        obj_qp - obj_mp, 100*(obj_qp - obj_mp)/obj_mp);
% ------------------------- 10. Save the result for Python  ------------------------------ %% 

save('dcopf_data.mat', ...      % one tidy MAT-file that Python can read
     'Aineq', 'Bineq', 'bbineq', ...
     'pd', ...
     'fval', ...
     'Aeq', 'Beq', ...
     'Q', 'c', ...
     'pg_opt');                 % Matlab solution to compare against
fprintf('Saved dcopf_data.mat\n');


