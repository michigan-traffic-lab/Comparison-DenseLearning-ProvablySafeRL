import time
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import gurobipy # dummy import gurobipy to make sure it's correctly installed
from casadi import *
from src.sets.Polytope import Polytope
from src.sets.PolyZonotope import PolyZonotope
from src.sets.LevelSet import LevelSet


class PlanningInfeasibleError(Exception):
    def __init__(self, message: str):
        print(message)


def safety_shield(alpha, reach, obstacles, solver='linear', grouptol=None):
    """project the control input from the reinforcement learning policy to the closest safe control input"""

    # check for collisions between the reachable set and the obstacles
    con_linear = []
    con_nonlinear = []

    for o in obstacles:
        if o.set.c.shape[0] > 1:
            pgon = o.set.polygon()
        for i in range(0, len(reach.occ)):
            if (o.time is None) or (o.time.intersects(reach.time[i])):
                if (o.set.c.shape[0] == 1 and o.set.intersects(reach.pgon[i].V)) or \
                                    (o.set.c.shape[0] > 1 and reach.pgon[i].intersects(pgon)):
                    tmp = constraints_linear(reach.occ[i], o.set, reach.ind)
                    con_linear.append(tmp)
                    tmp = constraints_nonlinear(reach.occ[i], o.set, reach.ind)
                    con_nonlinear.append(tmp)

    # check if the control input is safe (using nonlinear constraints)
    safe = True
    if np.min(alpha) < -1 or np.max(alpha) > 1:
        safe = False
    else:
        for ls in con_nonlinear:
            if not ls.contains(alpha):
                safe = False
                break

    # projection to the closest safe control input
    if safe:
        #print("Proposed action is safe!")
        return alpha, None, "safe"
    else:
        #print("Proposed unsafe action, correcting...")
        try:
            if solver == 'linear':

                satisfiable, con_linear = remove_unsatisfiable_linear(con_linear)
                if satisfiable:
                    if grouptol is not None:
                        con_linear = group_linear_constraints(con_linear, grouptol)
                    alpha_, opt_time = projection_mixed_integer_linear(alpha, con_linear)
                else:
                    alpha_ = None

            elif solver == 'quadratic':

                satisfiable, con_linear = remove_unsatisfiable_linear(con_linear)
                if satisfiable:
                    if grouptol is not None:
                        con_linear = group_linear_constraints(con_linear, grouptol)
                    alpha_, opt_time = projection_mixed_integer_quadratic(alpha, con_linear)
                else:
                    alpha_ = None

            elif solver == 'nonlinear':

                satisfiable, con_nonlinear = remove_unsatisfiable_nonlinear(con_nonlinear)
                if satisfiable:
                    if grouptol is not None:
                        con_nonlinear = group_nonlinear_constraints(con_nonlinear, grouptol)
                    alpha_ = projection_mixed_integer_nonlinear(alpha, con_nonlinear)
                else:
                    alpha_ = None
            else:
                raise Exception('Specified solver not available!')
        except:
            alpha_ = None

        if alpha_ is None:

            #print("Choosing random point")
            try:
                alpha_, opt_time  = find_feasible_point(con_nonlinear)
            except:
                pass

            if alpha_ is None:
                #raise PlanningInfeasibleError('Planner failed for find a feasible solution!')
                #print('Planner failed for find a feasible solution! Use unsafe solution instead.')
                return alpha, None, "infeasible"

        return alpha_, opt_time, "corrected"


def projection_mixed_integer_linear(alpha, con):
    """project alpha to the next closest point that satisfies the constraints"""

    # objective function
    n = len(alpha)
    x = cp.Variable((n, 1))
    t1 = cp.Variable((n, 1))
    t2 = cp.Variable((n, 1))
    objective = cp.Minimize(cp.sum_squares(t1 + t2 - alpha))

    # constraints
    const = [-np.ones((n, 1)) <= x, x <= np.ones((n, 1))]
    const.append(t1 - t2 == x)
    const.append(t1 >= np.zeros((n, 1)))
    const.append(t2 >= np.zeros((n, 1)))

    for c in con:

        if c.c.shape[0] == 1:

            const.append(c.c @ x <= c.d)

        else:

            la = cp.Variable(c.c.shape[0], integer=True)
            tmp = np.zeros((n, 1))

            for i in range(c.c.shape[0]):
                x_ = cp.Variable((n, 1))
                const.append(c.c[[i], :] @ x_ <= la[i] * c.d[i])
                const.append(-np.ones((n, 1))*la[i] <= x_)
                const.append(x_ <= la[i]*np.ones((n, 1)))
                tmp = tmp + x_

            const.append(tmp == x)
            const.append(sum(la) == 1)
            const.append(la <= np.ones((c.c.shape[0], )))
            const.append(la >= np.zeros((c.c.shape[0], )))

    # solve optimization problem
    prob = cp.Problem(objective, const)
    t1 = time.time()
    try:
        prob.solve(solver="GUROBI")
    except cp.error.SolverError:
        prob.solve(solver="SCIP", 
        scip_params={
                       "separating/cgmip/timelimit": 100,
                       "propagating/nlobbt/nlptimelimit": 100,
                       "heuristics/completesol/maxlpiter": 100,
                       "separating/convexproj/nlptimelimit": 100,
                       "separating/gauge/nlptimelimit": 100,
                       "limits/softtime": 100
                   }
        )

    return x.value, time.time()-t1

def projection_mixed_integer_quadratic(alpha, con):
    """project alpha to the next closest point that satisfies the constraints"""

    # objective function
    n = len(alpha)
    x = cp.Variable((n, 1))
    objective = cp.Minimize(cp.sum_squares(x - alpha))

    # constraints
    const = [-np.ones((n, 1)) <= x, x <= np.ones((n, 1))]

    for c in con:

        if c.c.shape[0] == 1:

            const.append(c.c @ x <= c.d)

        else:

            la = cp.Variable(c.c.shape[0], integer=True)
            tmp = np.zeros((n, 1))

            for i in range(c.c.shape[0]):
                x_ = cp.Variable((n, 1))
                const.append(c.c[[i], :] @ x_ <= la[i] * c.d[i])
                const.append(-np.ones((n, 1))*la[i] <= x_)
                const.append(x_ <= la[i]*np.ones((n, 1)))
                tmp = tmp + x_

            const.append(tmp == x)
            const.append(sum(la) == 1)
            const.append(la <= np.ones((c.c.shape[0], )))
            const.append(la >= np.zeros((c.c.shape[0], )))

    # solve optimization problem
    prob = cp.Problem(objective, const)
    t1 = time.time()
    prob.solve(solver="SCIP",
        scip_params={
                       "separating/cgmip/timelimit": 100,
                       "propagating/nlobbt/nlptimelimit": 100,
                       "heuristics/completesol/maxlpiter": 100,
                       "separating/convexproj/nlptimelimit": 100,
                       "separating/gauge/nlptimelimit": 100,
                       "limits/softtime": 100
                   }
        )

    return x.value, time.time()-t1

def projection_mixed_integer_nonlinear(alpha, con):
    """project alpha to the next closest point that satisfies the constraints"""

    # objective function
    n = len(alpha)
    x = MX.sym('x', n)

    discrete = []
    var = [x]
    x0 = alpha[:, 0].tolist()

    for i in range(n):
        discrete += [False]

    objective = 0

    for i in range(n):
        objective = objective + (x[i] - alpha[i]) ** 2

    # constraints
    lb = (-np.ones((n, ))).tolist()
    ub = np.ones((n, )).tolist()
    const = [x]

    cnt_la = 1
    cnt_x = 1

    for c in con:

        if c.b.shape[0] == 1:

            tmp = 0
            for j in range(c.E.shape[1]):
                tmp_ = c.a[0, j]
                for k in range(n):
                    if c.E[k, j] > 0:
                        tmp_ = tmp_ * (x[k, 0] ** c.E[k, j])
                tmp = tmp + tmp_
            const.append(tmp)
            lb += [-np.inf]
            ub += [c.b[0][0]]

        else:

            la = MX.sym('lamda_' + str(cnt_la), c.b.shape[0])
            var += [la]
            x0 += np.zeros((c.b.shape[0], )).tolist()
            for i in range(c.b.shape[0]):
                discrete += [True]
            cnt_la += 1

            summation = np.zeros((n, 1))

            for i in range(c.b.shape[0]):

                x_ = MX.sym('x_' + str(cnt_x), n)
                var += [x_]
                x0 += alpha[:, 0].tolist()
                for j in range(n):
                    discrete += [False]

                const += [x]
                lb += np.zeros((n, )).tolist()
                ub += np.ones((n, )).tolist()

                tmp = 0
                for j in range(c.E.shape[1]):
                    tmp_ = c.a[i, j]
                    for k in range(n):
                        if c.E[k, j] > 0:
                            tmp_ = tmp_ * (x_[k] ** c.E[k, j])
                    tmp = tmp + tmp_

                const += [tmp]
                lb += [-np.inf]
                ub += [c.b[i][0]]

                summation = summation + la[i] * x_

            const += [summation - x]
            lb += np.zeros((n, )).tolist()
            ub += np.zeros((n, )).tolist()

            const += [sum1(la)]
            lb += [1]
            ub += [1]

            const += [la]
            lb += np.zeros((c.b.shape[0], )).tolist()
            ub += np.ones((c.b.shape[0], )).tolist()

    # solve optimization problem
    nlp_prob = {'f': objective, 'x': vertcat(*var), 'g': vertcat(*const)}
    options = {'time_limit': 10, 'max_iter': 200, 'solution_limit': 1}
    nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete, 'bonmin': options})

    sol = nlp_solver(x0=x0, lbg=lb, ubg=ub)

    if nlp_solver.stats()['success']:
        return np.asarray(sol['x'][0:n])
    else:
        return None

def constraints_linear(pZ, poly, index):
    """extract linear constraints on the control commands from the reachable set"""

    # enclose all factors that do not belong to inputs with a zonotope
    ind = [i for i in range(pZ.E.shape[1]) if np.sum(pZ.E[index, i], axis=0) == 1 and np.sum(pZ.E[:, i], axis=0) == 1]
    ind_ = np.setdiff1d(np.arange(0, pZ.E.shape[1]), ind)

    Z = PolyZonotope(pZ.c, pZ.G[:, ind_], pZ.E[:, ind_], pZ.Grest).zonotope()

    E = pZ.E[:, ind]
    G = pZ.G[:, ind]

    # bring factors to the correct order
    G_= np.zeros((len(pZ.c), len(index)))

    for i in range(len(ind)):
        tmp = np.where(E[index[i], :] == 1)
        if len(tmp[0]) > 0:
            G_[:, [i]] = G[:, tmp[0]]

    # construct polytope constraints
    a = np.dot(-poly.c, G_)
    b = np.dot(poly.c, Z.c) - np.resize(np.sum(np.abs(np.dot(poly.c, Z.G)), axis=1), poly.d.shape) - poly.d

    if len(a.shape) == 1:
        a = np.resize(a, (1, a.shape[0]))

    return Polytope(a, b)

def constraints_nonlinear(pZ, poly, index):
    """extract nonlinear constraints on the control commands from the reachable set"""

    # enclose all factors that do not belong to inputs with a zonotope
    index_ = np.setdiff1d(np.arange(0, pZ.E.shape[0]), index)
    ind = np.where(np.sum(pZ.E[index_, :], axis=0) == 0)[0]
    ind_ = np.setdiff1d(np.arange(0, pZ.E.shape[1]), ind)

    Z = PolyZonotope(pZ.c, pZ.G[:, ind_], pZ.E[:, ind_], pZ.Grest).zonotope()

    # construct level set constraints
    a = -np.dot(poly.c, pZ.G[:, ind])
    b = np.dot(poly.c, Z.c) - np.expand_dims(np.sum(abs(np.dot(poly.c, Z.G)), axis=1), axis=1) - poly.d

    return LevelSet(a, b, pZ.E[:, ind][index, :])

def find_feasible_point(con_nonlinear):
    """find a feasible point that satisfies the constraints"""
    t1 = time.time()
    n = con_nonlinear[0].E.shape[0]
    alpha = np.random.uniform(-1, 1, (n, 1000))

    for i in range(alpha.shape[1]):

        safe = True

        for ls in con_nonlinear:
            if not ls.contains(alpha[:, [i]]):
                safe = False
                break

        if safe:
            return alpha[:, [i]], time.time()-t1

    return None, time.time()-t1

def remove_unsatisfiable_linear(con):
    """remove all constraints that cannot be satisfied from the or-connections"""

    con_new = []

    # loop over all constraints
    for c in con:

        # find constraints that are satisfiable on the domain \alpha \in [-1,1]
        tmp = -np.sum(abs(c.c), axis=1)
        ind = [i for i in range(len(tmp)) if tmp[i] <= c.d[i, 0]]

        # optimization problem has no solution if no satisfiable constraints are left
        if len(ind) == 0:
            return False, con_new

        # construct polytope from the satisfiable constraints
        con_new.append(Polytope(c.c[ind, :], c.d[ind, :]))

    return True, con_new


def remove_unsatisfiable_nonlinear(con):
    """remove all constraints that cannot be satisfied from the or-connections"""

    con_new = []

    # loop over all constraints
    for c in con:

        # determine indices of all-even exponents (because [-1,1]^e = [0,1] in this case)
        tmp = np.prod(np.ones(c.E.shape) - np.mod(c.E, 2), axis=0)
        ind = [i for i in range(0, len(tmp)) if tmp[i] == 1]
        ind_ = np.setdiff1d(np.arange(0, len(tmp)), ind)

        # find constraints that are satisfiable on the domain \alpha \in [-1,1]
        tmp = -np.sum(abs(c.a[:, ind_]), axis=1) - 0.5 * np.sum(abs(c.a[:, ind]), axis=1) \
              - 0.5 * np.sum(c.a[:, ind], axis=1)
        ind = [i for i in range(len(tmp)) if tmp[i] <= c.b[i, 0]]

        # optimization problem has no solution if no satisfiable constraints are left
        if len(ind) == 0:
            return False, con_new

        # construct level set from the satisfiable constraints
        con_new.append(LevelSet(c.a[ind, :], c.b[ind, :], c.E))

    return True, con_new


def group_linear_constraints(con, tol):
    """group together all constraints for which the over-approx. introduced by grouping is less than the tolerance"""

    con_grouped = []
    cnt = 0

    # normalize the length of the polytope halfspace normal vectors
    for i in range(len(con)):
        con[i] = con[i].normalize()

    # loop over the list of constraints and try to unite neighboring ones
    while cnt < len(con):

        # initialization
        cnt_unite = cnt+1
        unite = True
        c = con[cnt].c
        d = con[cnt].d

        # loop over neighbouring constraints until they cannot be united within the tolerance
        while cnt_unite < len(con):

            # only or-connections of constraints with identical size can be united
            if not np.all(con[cnt].c.shape == con[cnt_unite].c.shape):
                unite = False

            else:

                # compute averaged constraint
                c_mean = np.mean([con[j].c for j in range(cnt, cnt_unite+1)], axis=0)
                d_mean = np.ones((con[cnt].c.shape[0])) * np.inf

                # compute difference from the averaged constraint
                for i in range(cnt, cnt_unite+1):
                    c_diff = con[i].c - c_mean
                    d_mean = np.minimum(d_mean, np.squeeze(con[i].d) - np.sum(abs(c_diff), axis=1))

                d_mean = np.expand_dims(d_mean, axis=1)

                # check if the difference is within the tolerance for all constraints
                for i in range(cnt, cnt_unite+1):
                    if np.any(np.abs(con[i].d - d_mean) > tol):
                        unite = False
                        break

            # check if it was possible to unite the constraints
            if not unite:
                break
            else:
                c = c_mean
                d = d_mean
                cnt_unite = cnt_unite + 1

        # save the grouped constraint
        con_grouped.append(Polytope(c, d))
        cnt = cnt_unite

    return con_grouped


def group_nonlinear_constraints(con, tol):
    """group together all constraints for which the over-approx. introduced by grouping is less than the tolerance"""

    con_grouped = []
    cnt = 0

    # normalize the length of the coefficient vectors
    for i in range(len(con)):
        con[i] = con[i].normalize()

    # loop over the list of constraints and try to unite neighboring ones
    while cnt < len(con):

        # initialization
        cnt_unite = cnt + 1
        unite = True
        a = con[cnt].a
        b = con[cnt].b
        E = con[cnt].E

        # determine indices of all-even exponents (because [-1,1]^e = [0,1] in this case)
        tmp = np.prod(np.ones(E.shape) - np.mod(E, 2), axis=0)
        ind = [i for i in range(0, len(tmp)) if tmp[i] == 1]
        ind_ = np.setdiff1d(np.arange(0, len(tmp)), ind)

        # loop over neighbouring constraints until they cannot be united within the tolerance
        while cnt_unite < len(con):

            # only or-connections of constraints with identical size can be united
            if not np.all(con[cnt].a.shape == con[cnt_unite].a.shape) or \
                    not np.all(con[cnt].E.shape == con[cnt_unite].E.shape) or \
                    not np.all(con[cnt].E == con[cnt_unite].E):
                unite = False

            else:

                # compute averaged constraint
                a_mean = np.mean([con[j].a for j in range(cnt, cnt_unite + 1)], axis=0)
                b_mean = np.ones((con[cnt].a.shape[0])) * np.inf

                # compute difference from the averaged constraint
                for i in range(cnt, cnt_unite + 1):
                    a_diff = con[i].a - a_mean
                    tmp = np.sum(abs(a_diff[:, ind_]), axis=1) + 0.5 * np.sum(abs(a_diff[:, ind]), axis=1) \
                          + 0.5 * np.sum(a_diff[:, ind], axis=1)
                    b_mean = np.minimum(b_mean, np.squeeze(con[i].b) - tmp)

                b_mean = np.expand_dims(b_mean, axis=1)

                # check if the difference is within the tolerance for all constraints
                for i in range(cnt, cnt_unite + 1):
                    if np.any(np.abs(con[i].b - b_mean) > tol):
                        unite = False
                        break

            # check if it was possible to unite the constraints
            if not unite:
                break
            else:
                a = a_mean
                b = b_mean
                cnt_unite = cnt_unite + 1

        # save the grouped constraint
        con_grouped.append(LevelSet(a, b, E))
        cnt = cnt_unite

    return con_grouped
