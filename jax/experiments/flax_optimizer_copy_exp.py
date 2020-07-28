import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

rng             = jax.random.PRNGKey(0)
rng, key1, key2 = jax.random.split(rng, 3)

n = 50
x = jnp.linspace(-5., 5.)
X = jax.random.uniform(key1, (n,), minval=-5., maxval=5.)
f = lambda x: 2.*x
Y = f(x)+jax.random.normal(key2, (n,))

class LinearRegression(flax.nn.Module):
    def apply(self, x):
        return flax.nn.Dense(x[..., None], features=1)[..., 0]

rng, key  = jax.random.split(rng)
_, params = LinearRegression.init(key, X)
model     = flax.nn.Model(LinearRegression, params)


optimizer_def = flax.optim.Adam(learning_rate=0.001)
optimizer     = optimizer_def.create(model)
train_steps   = 10000

def loss_fn(model):
    y_predicted = model(X)
    return jnp.square(Y-y_predicted).mean()

for i in range(train_steps):
    loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer  = optimizer.apply_gradient(grad)

    updated_model = optimizer.target
    #optimizer     = None
    #del optimizer

    optimizer = flax.optim.Adam(learning_rate=0.001).create(updated_model)
    # update optimizer
    #optimizer.replace(optimizer_def=flax.optim.Adam(learning_rate=0.001).create(trained_model))
    print("MSE :", loss)

trained_model = optimizer.target
print("Сургасан моделийн параметр :")
print(trained_model.params)
plt.plot(x, f(x))
plt.plot(x, trained_model(x))
plt.scatter(X, Y)
plt.show()

