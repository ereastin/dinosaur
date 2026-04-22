# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for generating conditions for the Held-Suarez test case.

This test case is based on

  Held, I. M., and M. J. Suarez, 1994: "A proposal for the intercomparison of
  the dynamical cores of atmospheric general circulation models."
  Bulletin of the American Meteorological Society, 75, 1825–1830.
"""
from typing import Any, Callable, Mapping, Sequence

from dinosaur import sigma_coordinates
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur.primitive_equations import div_sec_lat
from dinosaur import scales
from dinosaur import time_integration
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np
import functools

units = scales.units
Quantity = units.Quantity
Array = typing.Array

# Variable names used to match format in Held-Suarez paper.
# pylint: disable=invalid-name

class HeldSuarezForcingTracers(time_integration.ExplicitODE):
  """The Held-Suarez test problem specification."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      reference_temperature: typing.Array,
      p0: Quantity = 1e5 * units.pascal,
      sigma_b: Quantity = 0.7,
      kf: Quantity = 1 / (1 * units.day),
      ka: Quantity = 1 / (40 * units.day),
      ks: Quantity = 1 / (4 * units.day),
      minT: Quantity = 200 * units.degK,
      maxT: Quantity = 315 * units.degK,
      dTy: Quantity = 60 * units.degK,
      dThz: Quantity = 10 * units.degK,
  ):
    """Initialize Held-Suarez with tracers

    Args:
      coords: horizontal and vertical descritization.
      physics_specs: object holding physical constants and definition of custom
        units to use for initialization of the state.
      reference_temperature: horizontal reference temperature at all altitudes.
      p0: reference surface pressure.
      sigma_b: sigma level of effective planetary boundary layer.
      kf: coefficient of friction for Rayleigh drag.
      ka: coefficient of thermal relaxation in upper atmosphere.
      ks: coefficient of thermal relaxation at earth surface on the equator.
      minT: lower temperature bound of radiative equilibrium.
      maxT: upper temperature bound of radiative equilibrium.
      dTy: horizontal temperature variation of radiative equilibrium.
      dThz: vertical temperature variation of radiative equilibrium.
    """
    self.coords = coords
    self.physics_specs = physics_specs
    self.reference_temperature = reference_temperature  # so this is still in Kelvin..
    self.p0 = physics_specs.nondimensionalize(p0)
    self.sigma_b = sigma_b
    self.kf = physics_specs.nondimensionalize(kf)
    self.ka = physics_specs.nondimensionalize(ka)
    self.ks = physics_specs.nondimensionalize(ks)
    self.minT = physics_specs.nondimensionalize(minT)
    self.maxT = physics_specs.nondimensionalize(maxT)
    self.dTy = physics_specs.nondimensionalize(dTy)
    self.dThz = physics_specs.nondimensionalize(dThz)
    # Coordinates
    self.sigma = self.coords.vertical.centers
    _, sin_lat = self.coords.horizontal.nodal_mesh
    self.lat = np.arcsin(sin_lat)
    # Storage
    self.surface_precip = []

  def kv(self):
    kv_coeff = self.kf * (
        np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    )
    return kv_coeff[:, np.newaxis, np.newaxis]

  def kt(self):
    cutoff = np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    return self.ka + (self.ks - self.ka) * (
        cutoff[:, np.newaxis, np.newaxis] * np.cos(self.lat) ** 4
    )

  def equilibrium_temperature(self, nodal_surface_pressure):
    p_over_p0 = (
        self.sigma[:, np.newaxis, np.newaxis] * nodal_surface_pressure / self.p0
    )
    temperature = p_over_p0**self.physics_specs.kappa * (
        self.maxT
        - self.dTy * np.sin(self.lat) ** 2
        - self.dThz * jnp.log(p_over_p0) * np.cos(self.lat) ** 2
    )
    return jnp.maximum(self.minT, temperature)

  def saturation_specific_humidity(self, temperature, surface_pressure):
    ### TODO: turn this into a lookup table or check if they have one! maybe JAX GCM has one.?
    T0 = 273.15  # K 
    T = temperature - T0
    saturation_vapor_pressure = 610.94 * jnp.exp(17.625 * T / (T + 243.04)) * units.pascal  # saturation vapor pressure (Pa)
    pressure = self.physics_specs.dimensionalize(jnp.tile(
      self.coords.vertical.centers[:, np.newaxis, np.newaxis],
      self.coords.surface_nodal_shape
    ) * surface_pressure, units.pascal)  # this assumes p_top = 0
    eps = 0.622
    return eps * saturation_vapor_pressure / (pressure - (1 - eps) * saturation_vapor_pressure)

  def explicit_terms(
      self, state: primitive_equations.State
  ) -> primitive_equations.State:
    """Computes explicit tendencies due to Held-Suarez forcing."""
    aux_state = primitive_equations.compute_diagnostic_state(  # this is nodal state
        state=state, coords=self.coords
    )

    # Nodal velocity tendencies
    nodal_velocity_tendency = jax.tree.map(
        lambda x: -self.kv() * x / self.coords.horizontal.cos_lat**2,
        aux_state.cos_lat_u,
    )

    # Nodal temperature tendency
    nodal_temperature = (  # this is still in Kelvin
        self.reference_temperature[:, np.newaxis, np.newaxis]  # idk why is this still in temp units but...
        + aux_state.temperature_variation  # this must be K?
    )
    nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
        state.log_surface_pressure
    )
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)  # this is still non-dim version
    Teq = self.equilibrium_temperature(nodal_surface_pressure)  # this has no dimensions but magnitude is the exact same.!
    nodal_temperature_tendency = -self.kt() * (nodal_temperature - Teq)

    # Convert to modal
    temperature_tendency = self.coords.horizontal.to_modal(
        nodal_temperature_tendency
    )
    velocity_tendency = self.coords.horizontal.to_modal(nodal_velocity_tendency)
    vorticity_tendency = self.coords.horizontal.curl_cos_lat(velocity_tendency)
    divergence_tendency = self.coords.horizontal.div_cos_lat(velocity_tendency)

    # Zero log_surface_pressure tendency
    # PrimitiveEquations instance, which runs first, calcs some P_surf tendency, this just doesnt contribute anything extra
    log_surface_pressure_tendency = jnp.zeros_like(state.log_surface_pressure)

    ## TODO(ereastin): ok so primitive equations are 'composed' w/H-S forcing..
    ##  so do the 'physics' here; add sources and sinks, condensation, etc. since its always after advection step
    qsat = self.saturation_specific_humidity(nodal_temperature, nodal_surface_pressure).magnitude
    # nudge ~BL to saturation over 30 min (Ming & Held) assume time not exactly imp(?)
    # would making this smoother reduce ringing?
    # set bottom 30%?
    pbl_top = qsat.shape[0] * 3 // 10
    ## TODO: find a better way to set this based on integration step
    nodal_sources = qsat.at[:-pbl_top, :, :].set(0) / 30  # over 3 time steps it would saturate.?
    nudged_qv = aux_state.tracers['qv'] + nodal_sources
    condense = nudged_qv - qsat  # ok so have this, how would we keep track of how much is rained out.?
    nodal_sinks = jnp.maximum(condense, 0) # calc condensation above sat specific hum

    ## TODO(ereastin): issue with negative tracer values, probably from Gibbs ringing across the qv gradients
    ## 1. need Semi-Lagrangian advection (hard, see git issue)
    ## 2. post-fix by scan for negative values and borrow mass from neighbor cells, where would this go.?
    combined = nodal_sources - nodal_sinks
    qv_tendency = self.coords.horizontal.to_modal(combined)
    tracers_tendency = {'qv': qv_tendency}

    return primitive_equations.State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
     #   precip=precip_tendency,
        tracers=tracers_tendency,
    )


class HeldSuarezForcing(time_integration.ExplicitODE):
  """The Held-Suarez test problem specification."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      reference_temperature: typing.Array,
      p0: Quantity = 1e5 * units.pascal,
      sigma_b: Quantity = 0.7,
      kf: Quantity = 1 / (1 * units.day),
      ka: Quantity = 1 / (40 * units.day),
      ks: Quantity = 1 / (4 * units.day),
      minT: Quantity = 200 * units.degK,
      maxT: Quantity = 315 * units.degK,
      dTy: Quantity = 60 * units.degK,
      dThz: Quantity = 10 * units.degK,
  ):
    """Initialize Held-Suarez.

    Args:
      coords: horizontal and vertical descritization.
      physics_specs: object holding physical constants and definition of custom
        units to use for initialization of the state.
      reference_temperature: horizontal reference temperature at all altitudes.
      p0: reference surface pressure.
      sigma_b: sigma level of effective planetary boundary layer.
      kf: coefficient of friction for Rayleigh drag.
      ka: coefficient of thermal relaxation in upper atmosphere.
      ks: coefficient of thermal relaxation at earth surface on the equator.
      minT: lower temperature bound of radiative equilibrium.
      maxT: upper temperature bound of radiative equilibrium.
      dTy: horizontal temperature variation of radiative equilibrium.
      dThz: vertical temperature variation of radiative equilibrium.
    """
    self.coords = coords
    self.physics_specs = physics_specs
    self.reference_temperature = reference_temperature
    self.p0 = physics_specs.nondimensionalize(p0)
    self.sigma_b = sigma_b
    self.kf = physics_specs.nondimensionalize(kf)
    self.ka = physics_specs.nondimensionalize(ka)
    self.ks = physics_specs.nondimensionalize(ks)
    self.minT = physics_specs.nondimensionalize(minT)
    self.maxT = physics_specs.nondimensionalize(maxT)
    self.dTy = physics_specs.nondimensionalize(dTy)
    self.dThz = physics_specs.nondimensionalize(dThz)
    # Coordinates
    self.sigma = self.coords.vertical.centers
    _, sin_lat = self.coords.horizontal.nodal_mesh
    self.lat = np.arcsin(sin_lat)

  def kv(self):
    kv_coeff = self.kf * (
        np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    )
    return kv_coeff[:, np.newaxis, np.newaxis]

  def kt(self):
    cutoff = np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    return self.ka + (self.ks - self.ka) * (
        cutoff[:, np.newaxis, np.newaxis] * np.cos(self.lat) ** 4
    )

  def equilibrium_temperature(self, nodal_surface_pressure):
    p_over_p0 = (
        self.sigma[:, np.newaxis, np.newaxis] * nodal_surface_pressure / self.p0
    )
    temperature = p_over_p0**self.physics_specs.kappa * (
        self.maxT
        - self.dTy * np.sin(self.lat) ** 2
        - self.dThz * jnp.log(p_over_p0) * np.cos(self.lat) ** 2
    )
    return jnp.maximum(self.minT, temperature)

  def explicit_terms(
      self, state: primitive_equations.State
  ) -> primitive_equations.State:
    """Computes explicit tendencies due to Held-Suarez forcing."""
    aux_state = primitive_equations.compute_diagnostic_state(
        state=state, coords=self.coords
    )

    # Nodal velocity tendencies
    nodal_velocity_tendency = jax.tree.map(
        lambda x: -self.kv() * x / self.coords.horizontal.cos_lat**2,
        aux_state.cos_lat_u,
    )

    # Nodal temperature tendency
    nodal_temperature = (
        self.reference_temperature[:, np.newaxis, np.newaxis]
        + aux_state.temperature_variation
    )
    nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
        state.log_surface_pressure
    )
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
    Teq = self.equilibrium_temperature(nodal_surface_pressure)
    nodal_temperature_tendency = -self.kt() * (nodal_temperature - Teq)

    # Convert to modal
    temperature_tendency = self.coords.horizontal.to_modal(
        nodal_temperature_tendency
    )
    velocity_tendency = self.coords.horizontal.to_modal(nodal_velocity_tendency)
    vorticity_tendency = self.coords.horizontal.curl_cos_lat(velocity_tendency)
    divergence_tendency = self.coords.horizontal.div_cos_lat(velocity_tendency)

    # Zero log_surface_pressure tendency
    log_surface_pressure_tendency = jnp.zeros_like(state.log_surface_pressure)

    return primitive_equations.State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
    )
