from __future__ import division
from __future__ import print_function
import os
import numpy as np
import fidimag.extensions.clib as clib
import fidimag.extensions.cvode as cvode
import fidimag.common.helper as helper
from fidimag.common.fileio import DataSaver
from fidimag.common.save_vtk import SaveVTK
import fidimag.common.constant as const


class LLG(object):

    def __init__(self, mesh, name='unnamed', use_jac=False, num_threads=1):
        """Simulation object.
        *Arguments*
          name : the Simulation name (used for writing data files, for examples)
        """
        self.num_threads = num_threads
        self.t = 0
        self.name = name
        self.mesh = mesh
        self.n = mesh.n
        self.n_nonzero = mesh.n
        self.unit_length = mesh.unit_length
        self._alpha = np.zeros(self.n, dtype=np.float)
        self._mu_s = np.zeros(self.n, dtype=np.float)
        self._mu_s_inv = np.zeros(self.n, dtype=np.float)

        self.spin = np.ones(3 * self.n, dtype=np.float)
        self.spin_last = np.ones(3 * self.n, dtype=np.float)
        self._pins = np.zeros(self.n, dtype=np.int32)
        self.field = np.zeros(3 * self.n, dtype=np.float)
        self.dm_dt = np.zeros(3 * self.n, dtype=np.float)
        self._skx_number = np.zeros(self.n, dtype=np.float)

        self.interactions = []
        self.pin_fun = None

        self.step = 0

        self.saver = DataSaver(self, name + '.txt')

        self.saver.entities['E_total'] = {
            'unit': '<J>',
            'get': lambda sim: sim.compute_energy(),
            'header': 'E_total'}

        self.saver.entities['m_error'] = {
            'unit': '<>',
            'get': lambda sim: sim.compute_spin_error(),
            'header': 'm_error'}

        self.saver.entities['skx_num'] = {
            'unit': '<>',
            'get': lambda sim: sim.skyrmion_number(),
            'header': 'skx_num'}

        self.saver.update_entity_order()

        # This is only for old C files using the xperiodic variable
        self.xperiodic, self.yperiodic = mesh.periodicity[0], mesh.periodicity[1]

        self.vtk = SaveVTK(self.mesh, name=name)

        if use_jac is not True:
            self.vode = cvode.CvodeSolver(self.spin, self.sundials_rhs, self.num_threads)
        else:
            self.vode = cvode.CvodeSolver(
                self.spin, self.sundials_rhs, self.sundials_jtn, self.num_threads)

        self.set_default_options()

        self.set_tols()

        # When initialising the integrator in the self.vode call, the CVOde
        # class calls the set_initial_value function (with flag_m=0), which
        # initialises a new integrator and allocates memory in this process.
        # Now, when we set the magnetisation, we will use the same memory
        # setting this flag_m to 1, so instead of calling CVodeInit we call
        # CVodeReInit. If don't, memory is allocated in every call of set_m
        self.flag_m = 1

    def set_default_options(self, gamma=1, mu_s=1, alpha=0.1):
        self.default_c = -1
        self._alpha[:] = alpha
        self._mu_s[:] = mu_s
        self.gamma = gamma
        self.do_precession = True

    def reset_integrator(self, t=0):
        self.vode.reset(self.spin, t)
        self.t = t  # also reinitialise the simulation time and step
        self.step = 0

    def set_tols(self, rtol=1e-8, atol=1e-10):
        self.vode.set_options(rtol, atol)

    def set_options(self, rtol=1e-8, atol=1e-10):
        self.set_tols(rtol, atol)

    def set_m(self, m0=(1, 0, 0), normalise=True):

        self.spin[:] = helper.init_vector(m0, self.mesh, normalise)

        # TODO: carefully checking and requires to call set_mu first
        self.spin.shape = (-1, 3)
        for i in range(self.spin.shape[0]):
            if self._mu_s[i] == 0:
                self.spin[i, :] = 0
        self.spin.shape = (-1,)

        self.vode.set_initial_value(self.spin, self.t)

        if not self.flag_m:
            self.flag_m = 1

    def get_pins(self):
        return self._pins

    def set_pins(self, pin):
        self._pins[:] = helper.init_scalar(pin, self.mesh)

        for i in range(len(self._mu_s)):
            if self._mu_s[i] == 0.0:
                self._pins[i] = 1

    pins = property(get_pins, set_pins)

    def get_alpha(self):
        return self._alpha

    def set_alpha(self, value):
        self._alpha[:] = helper.init_scalar(value, self.mesh)

    alpha = property(get_alpha, set_alpha)

    def get_mu_s(self):
        return self._mu_s

    def set_mu_s(self, value):
        self._mu_s[:] = helper.init_scalar(value, self.mesh)
        nonzero = 0
        for i in range(self.n):
            if self._mu_s[i] > 0.0:
                self._mu_s_inv[i] = 1.0 / self._mu_s[i]
                nonzero += 1

        self.n_nonzero = nonzero

        for i in range(len(self._mu_s)):
            if self._mu_s[i] == 0.0:
                self._pins[i] = 1

    mu_s = property(get_mu_s, set_mu_s)

    def add(self, interaction, save_field=False):
        interaction.setup(self.mesh, self.spin, self._mu_s)

        # TODO: FIX
        for i in self.interactions:
            if i.name == interaction.name:
                interaction.name = i.name + '_2'

        self.interactions.append(interaction)

        energy_name = 'E_{0}'.format(interaction.name)
        self.saver.entities[energy_name] = {
            'unit': '<J>',
            'get': lambda sim: sim.get_interaction(interaction.name).compute_energy(),
            'header': energy_name}

        if save_field:
            fn = '{0}'.format(interaction.name)
            self.saver.entities[fn] = {
                'unit': '<>',
                'get': lambda sim: sim.get_interaction(interaction.name).average_field(),
                'header': ('%s_x' % fn, '%s_y' % fn, '%s_z' % fn)}

        self.saver.update_entity_order()

    def get_interaction(self, name):
        for interaction in self.interactions:
            if interaction.name == name:
                return interaction
        else:
            raise ValueError("Failed to find the interaction with name '{0}', "
                             "available interactions: {1}.".format(
                                 name, [x.name for x in self.interactions]))

    def run_until(self, t):

        if t <= self.t:
            if t == self.t and self.t == 0.0:
                self.compute_effective_field(t)
                self.saver.save()
            return

        ode = self.vode

        self.spin_last[:] = self.spin[:]

        flag = ode.run_until(t)

        if flag < 0:
            raise Exception("Run cython run_until failed!!!")

        self.spin[:] = ode.y[:]

        self.t = t
        self.step += 1

        # update field before saving data
        self.compute_effective_field(t)
        self.saver.save()

    def compute_effective_field(self, t):

        #self.spin[:] = y[:]

        self.field[:] = 0

        for obj in self.interactions:
            self.field += obj.compute_field(t)

    def compute_effective_field_jac(self, t, spin):

        #self.spin[:] = y[:]

        self.field[:] = 0

        for obj in self.interactions:
            if obj.jac is True:
                self.field += obj.compute_field(t, spin=spin)

    def sundials_rhs(self, t, y, ydot):

        self.t = t

        # already synchronized when call this funciton
        # self.spin[:]=y[:]

        self.compute_effective_field(t)

        clib.compute_llg_rhs(ydot,
                             self.spin,
                             self.field,
                             self.alpha,
                             self._pins,
                             self.gamma,
                             self.n,
                             self.do_precession,
                             self.default_c)

        #ydot[:] = self.dm_dt[:]

        return 0

    def sundials_jtn(self, mp, Jmp, t, m, fy):
        # we can not copy mp to self.spin since m and self.spin is one object.
        #self.spin[:] = mp[:]
        print('NO jac...........')
        self.compute_effective_field_jac(t, mp)
        clib.compute_llg_jtimes(Jmp,
                                m, fy,
                                mp, self.field,
                                self.alpha,
                                self._pins,
                                self.gamma,
                                self.n,
                                self.do_precession,
                                self.default_c)

        return 0

    def compute_average(self):
        self.spin.shape = (-1, 3)
        average = np.sum(self.spin, axis=0) / self.n_nonzero
        self.spin.shape = (-1,)
        return average

    def compute_energy(self):

        energy = 0

        for obj in self.interactions:
            energy += obj.compute_energy()

        return energy

    def skyrmion_number(self):
        nx = self.mesh.nx
        ny = self.mesh.ny
        nz = self.mesh.nz
        number = clib.compute_skyrmion_number(
            self.spin, self._skx_number, nx, ny, nz, self.mesh.neighbours)
        return number

    def spin_at(self, i, j, k):
        """
        Returns the spin components of the spin at
        i-th position in the z direction, j-th position in the
        y direction and k-th position in x direction
        """

        nxyz = self.mesh.n

        index = 3 * self.mesh.index(i, j, k)

        # print self.spin.shape,nxy,nx,i1,i2,i3
        return np.array([self.spin[index],
                         self.spin[index + 1],
                         self.spin[index + 2]])

    def add_monitor_at(self, i, j, k, name='p'):
        """
        Save site spin with index (i,j,k) to txt file.
        """

        self.saver.entities[name] = {
            'unit': '<>',
            'get': lambda sim: sim.spin_at(i, j, k),
            'header': (name + '_x', name + '_y', name + '_z')}

        self.saver.update_entity_order()

    def save_vtk(self):
        """
        Save a VTK file with the magnetisation vector field and magnetic
        moments as cell data. Magnetic moments are saved in units of
        Bohr magnetons

        NOTE: It is recommended to use a *cell to point data* filter in
        Paraview or Mayavi to plot the vector field
        """
        self.vtk.save_vtk(self.spin.reshape(-1, 3),
                          self._mu_s / const.mu_B,
                          step=self.step
                          )

    def save_m(self):
        if not os.path.exists('%s_npys' % self.name):
            os.makedirs('%s_npys' % self.name)
        name = '%s_npys/m_%g.npy' % (self.name, self.step)
        np.save(name, self.spin)

    def save_skx(self, vtk=False):
        if not os.path.exists('%s_skx_npys' % self.name):
            os.makedirs('%s_skx_npys' % self.name)
        name = '%s_skx_npys/m_%g.npy' % (self.name, self.step)
        np.save(name, self._skx_number)
        if vtk is True:
            self.vtk.save_vtk_scalar(self._skx_number, self.step)

    def stat(self):
        return self.vode.stat()

    def spin_length(self):
        self.spin.shape = (-1, 3)
        length = np.sqrt(np.sum(self.spin**2, axis=1))
        self.spin.shape = (-1,)
        return length

    def compute_spin_error(self):
        length = self.spin_length() - 1.0
        length[self._pins > 0] = 0
        return np.max(abs(length))

    def compute_dmdt(self, dt):
        m0 = self.spin_last
        m1 = self.spin
        dm = (m1 - m0).reshape((-1, 3))
        max_dm = np.max(np.sqrt(np.sum(dm**2, axis=1)))
        max_dmdt = max_dm / dt
        return max_dmdt

    def relax(self, dt=1e-11, stopping_dmdt=0.01,
              max_steps=1000, save_m_steps=100, save_vtk_steps=100):

        for i in range(0, max_steps + 1):

            cvode_dt = self.vode.get_current_step()

            increment_dt = dt

            if cvode_dt > dt:
                increment_dt = cvode_dt

            self.run_until(self.t + increment_dt)

            if save_vtk_steps is not None:
                if i % save_vtk_steps == 0:
                    self.save_vtk()
            if save_m_steps is not None:
                if i % save_m_steps == 0:
                    self.save_m()

            dmdt = self.compute_dmdt(increment_dt)

            print('step=%d, time=%0.3g, max_dmdt=%0.3g ode_step=%0.3g' % (self.step,self.t, dmdt,cvode_dt))

            if dmdt < stopping_dmdt:
                break

        if save_m_steps is not None:
            self.save_m()

        if save_vtk_steps is not None:
            self.save_vtk()


if __name__ == '__main__':
    pass
