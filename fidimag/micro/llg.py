from __future__ import division
from __future__ import print_function
import os
import time
import fidimag.extensions.clib as clib
import fidimag.extensions.micro_clib as micro_clib
import numpy as np
from fidimag.common.fileio import DataSaver, DataReader
from fidimag.common.save_vtk import SaveVTK
from fidimag.common.integrators import SundialsIntegrator, StepIntegrator
import fidimag.common.constant as const
import fidimag.common.helper as helper
import re


class LLG(object):

    def __init__(self, mesh, name='unnamed', integrator='sundials', use_jac=False, num_threads=1):
        """Simulation object.

        *Arguments*

          name : the Simulation name (used for writing data files, for examples)

        """
        self.num_threads=num_threads
        self.t = 0
        self.name = name
        self.mesh = mesh
        self.n = mesh.n
        self.n_nonzero = mesh.n
        self.unit_length = mesh.unit_length
        self._alpha = np.zeros(self.n, dtype=np.float)
        self._Ms = np.zeros(self.n, dtype=np.float)
        self._Ms_inv = np.zeros(self.n, dtype=np.float)
        self.spin = np.ones(3 * self.n, dtype=np.float)
        self.spin_last = np.ones(3 * self.n, dtype=np.float)
        self._pins = np.zeros(self.n, dtype=np.int32)
        self.field = np.zeros(3 * self.n, dtype=np.float)
        self.dm_dt = np.zeros(3 * self.n, dtype=np.float)
        self._skx_number = np.zeros(self.n, dtype=np.float)
        self.interactions = []
        self.integrator_tolerances_set = False
        self.step = 0

        if integrator == "sundials" and use_jac:
            self.integrator = SundialsIntegrator(self.spin, self.sundials_rhs, self.sundials_jtimes)
        elif integrator == "sundials_diag":
            self.integrator = SundialsIntegrator(self.spin, self.sundials_rhs, linear_solver="diag", num_threads=self.num_threads)
        elif integrator == "sundials":
            self.integrator = SundialsIntegrator(self.spin, self.sundials_rhs, num_threads=self.num_threads)
        elif integrator == "euler" or integrator == "rk4":
            self.integrator = StepIntegrator(self.spin, self.step_rhs, integrator)
        else:
            raise NotImplemented("integrator must be sundials, euler or rk4")

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

        self.saver.entities['rhs_evals'] = {
            'unit': '<>',
            'get': lambda sim: self.integrator.rhs_evals(),
            'header': 'rhs_evals'}

        self.saver.entities['real_time'] = {
            'unit': '<s>',
            'get': lambda _: time.time(),  # seconds since epoch
            'header': 'real_time'}

        self.saver.update_entity_order()

        # This is for old C files codes using the xperiodic variables
        self.xperiodic, self.yperiodic, self.zperiodic = mesh.periodicity

        self.vtk = SaveVTK(self.mesh, name=name)

        self.set_default_options()

    def set_default_options(self, gamma=2.21e5, Ms=8.0e5, alpha=0.1):
        self.default_c = 1e11
        self._alpha[:] = alpha
        self._Ms[:] = Ms
        self.gamma = gamma
        self.do_precession = True

    def reset_integrator(self, t=0):
        self.integrator.reset(self.spin, t)
        self.t = t # also reinitialise the simulation time and step
        self.step = 0

    def set_tols(self, rtol=1e-8, atol=1e-10, max_ord=None, reset=True):
        if max_ord is not None:
            self.integrator.set_options(rtol=rtol, atol=atol, max_ord=max_ord)
        else:
            # not all integrators have max_ord (only VODE does)
            # and we don't want to encode a default value here either
            self.integrator.set_options(rtol=rtol, atol=atol)
        if reset:
            self.reset_integrator(self.t)

    def set_m(self, m0=(1, 0, 0), normalise=True):

        self.spin[:] = helper.init_vector(m0, self.mesh, normalise)

        # TODO: carefully checking and requires to call set_mu first
        self.spin.shape = (-1, 3)
        for i in range(self.spin.shape[0]):
            if self._Ms[i] == 0:
                self.spin[i, :] = 0
        self.spin.shape = (-1,)

        self.integrator.set_initial_value(self.spin, self.t)

    def get_pins(self):
        return self._pins

    def set_pins(self, pin):
        self._pins[:] = helper.init_scalar(pin, self.mesh)

        for i in range(len(self._Ms)):
            if self._Ms[i] == 0.0:
                self._pins[i] = 1

    pins = property(get_pins, set_pins)

    def get_alpha(self):
        return self._alpha

    def set_alpha(self, value):
        self._alpha[:] = helper.init_scalar(value, self.mesh)

    alpha = property(get_alpha, set_alpha)

    def get_Ms(self):
        return self._Ms

    def set_Ms(self, value):
        self._Ms[:] = helper.init_scalar(value, self.mesh)
        nonzero = 0
        for i in range(self.n):
            if self._Ms[i] > 0.0:
                self._Ms_inv = 1.0 / self._Ms[i]
                nonzero += 1

        self.n_nonzero = nonzero

        for i in range(len(self._Ms)):
            if self._Ms[i] == 0.0:
                self._pins[i] = 1

        self.Ms_const = np.max(self._Ms)

    Ms = property(get_Ms, set_Ms)

    def add(self, interaction, save_field=False):
        interaction.setup(self.mesh, self.spin, Ms=self._Ms)

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

        self.spin_last[:] = self.spin[:]

        flag = self.integrator.run_until(t)
        if flag < 0:
            raise Exception("Run run_until failed!!!")

        self.spin[:] = self.integrator.y[:]
        self.t = t
        self.step += 1

        self.compute_effective_field(t) # update fields before saving data
        self.saver.save()

    def compute_effective_field(self, t):

        #self.spin[:] = y[:]

        self.field[:] = 0

        for obj in self.interactions:
            self.field += obj.compute_field(t)

    def compute_effective_field_jac(self, t, spin):
        self.field[:] = 0
        for obj in self.interactions:
            if obj.jac:
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

    def sundials_jtimes(self, mp, Jmp, t, m, fy):
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

    def step_rhs(self, t, y):
        self.spin[:] = y[:]
        self.t = t
        self.compute_effective_field(t)
        clib.compute_llg_rhs(self.dm_dt,
                             self.spin,
                             self.field,
                             self.alpha,
                             self._pins,
                             self.gamma,
                             self.n,
                             self.do_precession,
                             self.default_c)
        return self.dm_dt

    def compute_average(self):
        self.spin.shape = (-1, 3)
        average = np.sum(self.spin, axis=0) / self.n_nonzero
        self.spin.shape = (3 * self.n)
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
        number = micro_clib.compute_skyrmion_number(
            self.spin, self._skx_number, nx, ny, nz, self.mesh.neighbours)
        return number

    def spin_at(self, i, j, k):

        i1 = 3 * self.mesh.index(i, j, k)

        # print self.spin.shape,nxy,nx,i1,i2,i3
        return np.array([self.spin[i1],
                         self.spin[i1 + 1],
                         self.spin[i1 + 2]])

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
        self.vtk.save_vtk(self.spin.reshape(-1, 3), self.Ms, step=self.step)

    def save_m(self):
        if not os.path.exists('%s_npys' % self.name):
            os.makedirs('%s_npys' % self.name)
        name = '%s_npys/m_%g.npy' % (self.name, self.step)
        np.save(name, self.spin)

    def save_skx(self):
        if not os.path.exists('%s_skx_npys' % self.name):
            os.makedirs('%s_skx_npys' % self.name)
        name = '%s_skx_npys/m_%g.npy' % (self.name, self.step)
        np.save(name, self._skx_number)

    def stat(self):
        return self.integrator.stat()

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
        dm = (m1 - m0).reshape((3, -1))
        max_dm = np.max(np.sqrt(np.sum(dm**2, axis=0)))
        max_dmdt = max_dm / dt
        return max_dmdt

    def relax(self, dt=1e-11, stopping_dmdt=0.01, max_steps=1000,
              save_m_steps=100, save_vtk_steps=100):

        ONE_DEGREE_PER_NS = 17453292.52

        for i in range(0, max_steps + 1):

            cvode_dt = self.integrator.get_current_step()

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

            print('step=%d, time=%0.3g, max_dmdt=%0.3g ode_step=%0.3g' % (self.step,
                                                                 self.t,
                                                                 dmdt / ONE_DEGREE_PER_NS,
                                                                 cvode_dt))

            if dmdt < stopping_dmdt * ONE_DEGREE_PER_NS:
                break

        if save_m_steps is not None:
            self.save_m()

        if save_vtk_steps is not None:
            self.save_vtk()

if __name__ == '__main__':
    pass
