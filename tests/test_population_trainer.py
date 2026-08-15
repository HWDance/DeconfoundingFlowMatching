import torch

from deconfoundingfm.core.target import DeconfoundingFlow, DeconfoundingFlowConfig
from deconfoundingfm.datasets import ColorMNISTConfig, ColorMNISTPopulation
from deconfoundingfm.experimental import PopulationFlowTrainer, PopulationTargetConfig
from deconfoundingfm.nn.velocity import UNet
from deconfoundingfm.nuisance.outcome import ConditionalFlowFM, ConditionalFlowFMConfig


def _source(size=8):
    one = torch.zeros(4, 1, size, size)
    one[:, :, 1:-1, size // 2] = 1.0
    six = torch.zeros(4, 1, size, size)
    six[:, :, 1:-1, 1] = 1.0
    six[:, :, -2, 1:-2] = 1.0
    return ColorMNISTPopulation(
        {1: one, 6: six},
        config=ColorMNISTConfig(digits=(1, 6), confounding_w=2.0),
        device="cpu",
    )


def test_population_trainer_tiny_image_smoke():
    torch.manual_seed(0)
    source = _source()
    trainer = PopulationFlowTrainer(source)
    propensity = source.oracle_propensity()

    nuisance = ConditionalFlowFM(
        ConditionalFlowFMConfig(
            dim_y=1,
            dim_x=1,
            lr=1e-3,
            batch_size=2,
            ode_steps=1,
            base_kind="gaussian",
            velocity_kind="unetx",
            y_is_image=True,
            y_channels=3,
            y_height=8,
            y_width=8,
            num_classes=2,
            x_dim=1,
            unet_c=2,
            film_hidden=2,
        ),
        device="cpu",
    )
    trainer.fit_outcome(
        nuisance,
        iterations=1,
        batch_size=2,
        verbose=False,
    )

    target = DeconfoundingFlow(
        DeconfoundingFlowConfig(
            dim_y=1,
            base_kind="empirical",
            batch_size=2,
            lr=1e-3,
            ode_steps=1,
            plugin_reservoir=1,
            plugin_batch=1,
            use_ot=False,
        ),
        nuisance_outcome=nuisance,
        nuisance_pi=propensity,
        device="cpu",
        velocity=UNet(in_channels=3, out_channels=3, num_classes=2, c=2),
    )
    trainer.fit_target(
        target,
        iterations=1,
        batch_size=2,
        population=PopulationTargetConfig(
            context_reservoir_size=2,
            context_refresh_steps=1,
            plugin_context_chunk=2,
            plugin_ode_steps=1,
            reservoir_dtype="float32",
            amp=False,
        ),
        verbose=False,
    )
    samples = trainer.sample_target(target, 1, 2, chunk_size=2, ode_steps=1)
    assert samples.shape == (2, 3, 8, 8)
    assert target.training_steps_ == 1
