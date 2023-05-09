from __future__ import annotations 

import gc ###about simulator 
import math
from pathlib import Path
from typing import Callable, List, Optional

import pyro
import torch
from diffeqtorch import DiffEq
from pyro import distributions as pdist
from pyro.distributions.transforms import PowerTransform

import sbibm  # noqa -- needed for setting sysimage path
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.decorators import lazy_property


class Repressilator(Task):
    def __init__(
        self,
        days: float = 20.0, 
        saveat: float = 0.1, 
        total_count: int = 1000,#total count of waht. in klotka volterra alsothis value.something different there
        summary: Optional[str] = "subsample",
    ):
        """Repressilator synthetic genetic circuit
        Inference is performed for twelve parameters 
        (these four for TetR, LacI and lambda CI):
        - Hill coefficient, n
        - Gene expression rate, alpha
        - Leaky expression, alpha_0
        - Degradation rate protein/mRNA ratio, beta
        Args:
            m_gene: Amount of mRNA
            p_gene: Amount of protein
            days: Number of days
            saveat: When to save during solving
            summary: Summaries to use #quee??
        References:
            [1]: https://www.nature.com/articles/35002125
        """
        self.dim_data_raw = int(6 * (days / saveat + 1)) 

        if summary is None:
            dim_data = self.dim_data_raw
        elif summary == "subsample":
            dim_data = 10 #need to change
        else:
            raise NotImplementedError
        self.summary = summary
        self.total_count = total_count

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000000,  # observation 1
            1000001,  # observation 2
            1000010,  # observation 3* #why the asterisk
            1000011,  # observation 4*
            1000004,  # observation 5
            1000005,  # observation 6
            1000006,  # observation 7
            1000013,  # observation 8*
            1000008,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=12,
            dim_data=dim_data,
            name=Path(__file__).parent.name,
            name_display="Repressilator",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )
        # Prior 
        self.prior_params = { #am i doing same values for all. if not do torch.tensor([param1,param2..])
            "low": -1* torch.ones((self.dim_parameters,)), # value i set is random
            "high": 1 * torch.ones((self.dim_parameters,)),
        }
        log_uniform_dist = pdist.Uniform(**self.prior_params).to_event(1) 
        power_transform = PowerTransform(10.0)
        self.prior_dist = pdist.TransformedDistribution(log_uniform_dist, power_transform)
        self.prior_dist.set_default_validate_args(False)

        """this is on lotka
          mu_p1 = -0.125
        mu_p2 = -3.0
        sigma_p = 0.5
        self.prior_params = {
            "loc": torch.tensor([mu_p1, mu_p2, mu_p1, mu_p2]),
            "scale": torch.tensor([sigma_p, sigma_p, sigma_p, sigma_p]),
        }

        """

        self.u0 = torch.tensor([0,0,0,0,0,0])
        self.tspan = torch.tensor([0.0, days])
        self.days = days
        self.saveat = saveat

    @lazy_property
    def de(self):
        return DiffEq(
            f=f"""
            function f(du,u,p,t)
                m_tet,p_tet,m_lac,p_lac,m_lam,p_lam = u
                n_tet, n_lac, n_lam,a_tet,a_lac,a_lam,a_0_tet,a_0_lac,a_0_lam,b_tet,b_lac,b_lam = p
                du[1] = -m_tet + (a_tet / (1 + p_lac ** n_lac)) + a_0_tet
                du[2] = -b_tet*(p_tet-m_tet)
                du[3] = -m_lac + (a_lac / (1 + p_lam ** n_lam)) + a_0_lac
                du[4] = -b_lac*(p_lac-m_lac)
                du[5] = -m_lam + (a_lam / (1 + p_tet ** n_tet)) + a_0_lam
                du[6] = -b_lam*(p_lam-m_lam)
            end
            """,
            saveat=self.saveat,
            debug=False,  # 5
        )

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [r"$n_tet$", r"$n_lac$", r"$n_lam$", r"$\alpha_tet$", 
                r"$\alpha_lac$",r"$\alpha_lam$",r"$\alpha_0_tet$",
                r"$\alpha_0_lac$",r"$\alpha_0_lam$",r"$\beta_tet$",
                r"$\beta_lac$",r"$\beta_lam$",]

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(
        self,
        max_calls: Optional[int] = None,
    ) -> Simulator:
        """Get function returning samples from simulator given parameters
        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget
        Return:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            us = []
            for num_sample in range(num_samples):
                u, t = self.de(self.u0, self.tspan, parameters[num_sample, :])

                if u.shape != torch.Size([6, int(self.dim_data_raw / 6)]):  
                    u = float("nan") * torch.ones((6, int(self.dim_data_raw / 6)))
                    u = u.double()

                us.append(u.reshape(1, 6, -1))
            us = torch.cat(us).float()  # num_parameters x 6 x (days/saveat + 1)

            idx_contains_nan = torch.where(
                torch.isnan(us.reshape(num_samples, -1)).any(axis=1)
            )[
                0
            ]  # noqa
            idx_contains_no_nan = torch.where(
                ~torch.isnan(us.reshape(num_samples, -1)).any(axis=1)
            )[
                0
            ]  # noqa

            if self.summary is None:
                return us

            elif self.summary == "subsample":
                data = float("nan") * torch.ones((num_samples, self.dim_data))
                if len(idx_contains_nan) == num_samples:
                    return data

                us = us[:, 1, ::17].reshape(num_samples, -1)  # Only use I population
                data[idx_contains_no_nan, :] = pyro.sample(
                    "data",
                    pdist.Binomial(
                        total_count=self.total_count,
                        probs=(us[idx_contains_no_nan, :] / self.N).clamp(0.0, 1.0),
                    ).to_event(1),
                )
                return data

            else:
                raise NotImplementedError

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data into multiple observations"""
        if self.summary is None:
            return data.reshape(-1, 3, int(self.dim_data / 3)) #the 3 again
        else:
            return data.reshape(-1, self.dim_data)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation
        Args:
            num_observation: Observation number
            num_samples: Number of samples to generate
            observation: Observed data, if None, will be loaded using `num_observation`
            kwargs: Passed to run_mcmc
        Returns:
            Samples from reference posterior
        """
        from sbibm.algorithms.pyro.mcmc import run as run_mcmc
        from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
        from sbibm.algorithms.pytorch.utils.proposal import get_proposal

        if num_observation is not None:
            initial_params = self.get_true_parameters(num_observation=num_observation)
        else:
            initial_params = None

        proposal_samples = run_mcmc(
            task=self,
            kernel="Slice",
            jit_compile=False,
            num_warmup=10_000,
            num_chains=1,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            initial_params=initial_params,
            automatic_transforms_enabled=True,
        )

        proposal_dist = get_proposal(
            task=self,
            samples=proposal_samples,
            prior_weight=0.1,
            bounded=True,
            density_estimator="flow",
            flow_model="nsf",
        )

        samples = run_rejection(
            task=self,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            batch_size=10_000,
            num_batches_without_new_max=1_000,
            multiplier_M=1.2,
            proposal_dist=proposal_dist,
        )

        return samples


if __name__ == "__main__":
    task = Repressilator()
    task._setup()
