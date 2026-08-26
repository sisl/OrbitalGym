import flax.struct
import jax
import jax.numpy as jnp

from orbitalgym.belief.contact_aware import ContactAwareBelief


class _StubBelief:
    """Minimal stub — only needs `mean` to satisfy the Belief protocol."""

    def __init__(self, mean):
        self.mean = mean


@flax.struct.dataclass
class _FlaxStubBelief:
    """flax-struct inner so the pytree threading test exercises real pytree leaves."""

    mean: jax.Array


def test_contact_aware_delegates_mean():
    inner = _StubBelief(mean=jnp.zeros((1, 2, 6)))
    cab = ContactAwareBelief(inner=inner, contact=jnp.asarray([True]))
    assert cab.mean.shape == (1, 2, 6)


def test_contact_aware_exposes_contact():
    inner = _StubBelief(mean=jnp.zeros((1, 2, 6)))
    cab = ContactAwareBelief(inner=inner, contact=jnp.asarray([True, False]))
    assert bool(cab.contact[0]) is True
    assert bool(cab.contact[1]) is False


def test_contact_aware_is_pytree():
    """Round-trip through tree_map — required for jax.lax.scan threading."""
    inner = _StubBelief(mean=jnp.zeros((1, 2, 6)))
    cab = ContactAwareBelief(inner=inner, contact=jnp.asarray([True]))
    leaves = jax.tree.leaves(cab)
    # The Stub isn't a pytree, so only `contact` is a leaf — the inner is
    # treated as a static node. We swap to a flax dataclass inner in real use.
    assert any(isinstance(leaf, jax.Array) for leaf in leaves)


def test_contact_aware_pytree_threading_through_scan():
    """Round-trip a ContactAwareBelief through jax.lax.scan to ensure the
    inner belief's leaves are actually threaded as pytree leaves."""
    inner = _FlaxStubBelief(mean=jnp.zeros((1, 2, 6), dtype=jnp.float32))
    cab = ContactAwareBelief(inner=inner, contact=jnp.asarray([True]))

    def body(carry, x):
        # Trivial body that increments inner.mean; if the pytree threading is
        # broken, this will fail to trace.
        new_cab = ContactAwareBelief(
            inner=_FlaxStubBelief(mean=carry.inner.mean + x),
            contact=carry.contact,
        )
        return new_cab, new_cab.inner.mean.sum()

    final, ys = jax.lax.scan(body, cab, jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32))
    assert ys.shape == (3,)
    assert float(final.inner.mean[0, 0, 0]) == 6.0
