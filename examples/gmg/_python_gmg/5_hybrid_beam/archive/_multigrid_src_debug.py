
# # compare the solution the fine solution to prolongate solution (I think something is wrong with the theta DOFs)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# ax[0,0].plot(grids[0].xvec, defects[i][0::2])
# ax[0,1].plot(grids[1].xvec, defects[i+1][0::2])
# ax[1,0].plot(grids[0].xvec, defects[i][1::2])
# ax[1,1].plot(grids[1].xvec, defects[i+1][1::2])
# plt.show()

# temp debug, plot part of the prolong correction process..
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# ax[0,0].plot(grids[0].xvec, dx[0::2])
# ax[0,1].plot(grids[0].xvec, defects[i][0::2])
# ax[1,0].plot(grids[0].xvec, dx[1::2])
# ax[1,1].plot(grids[0].xvec, defects[i][1::2])
# plt.show()

# prev fine defect to prolong defect
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# ax[0,0].plot(grids[0].xvec, -df[0::2])
# ax[0,1].plot(grids[0].xvec, defects[i][0::2])
# ax[1,0].plot(grids[0].xvec, -df[1::2])
# ax[1,1].plot(grids[0].xvec, defects[i][1::2])
# plt.show()


# plot the coarse defect and solve
        # i = nlevels-1
        # # idof = 0
        # idof = 1
        # grids[i].u = defects[i].copy()
        # grids[i].plot_disp(idof)
        # grids[i].u = solns[i].copy()
        # grids[i].plot_disp(idof)


# # compare the solution the fine solution to prolongate solution (I think something is wrong with the theta DOFs)
# import matplotlib.pyplot as plt
# fine_soln = sp.sparse.linalg.spsolve(mats[i].copy(), defects[i])
# fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# ax[0,0].plot(grids[0].xvec, dx[0::3])
# ax[0,1].plot(grids[0].xvec, fine_soln[0::3])
# ax[1,0].plot(grids[0].xvec, dx[1::3])
# ax[1,1].plot(grids[0].xvec, fine_soln[1::3])
# plt.show()

# # plot change in defect
# import matplotlib.pyplot as plt
# ddf = -omega * df
# fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# ax[0,0].plot(grids[0].xvec, ddf[0::2])
# ax[0,1].plot(grids[0].xvec, defect_init[0::2])
# ax[1,0].plot(grids[0].xvec, ddf[1::2])
# ax[1,1].plot(grids[0].xvec, defect_init[1::2])
# plt.show()

# # plot final in defect
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# ax[0,0].plot(grids[0].xvec, defect_init[0::2])
# ax[0,1].plot(grids[0].xvec, defects[i][0::2])
# ax[1,0].plot(grids[0].xvec, defect_init[1::2])
# ax[1,1].plot(grids[0].xvec, defects[i][1::2])
# plt.show()