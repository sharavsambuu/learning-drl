import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

rng = jax.random.PRNGKey(0)
rng, key1, key2 = jax.random.split(rng, 3)

n = 50
x = jnp.linspace(-5., 5., num=100)
X = jax.random.uniform(key1, (n, 1), minval=-5., maxval=5.)  # Reshape X to have a feature dimension
f = lambda x: 2. * x
Y = f(X) + jax.random.normal(key2, (n, 1))  # Adjust Y to match the shape of X
plt.plot(x, f(x), label='True function')
plt.scatter(X, Y, label='Data')
plt.legend()
plt.show()

class LinearRegression(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=1)(x)

rng, key = jax.random.split(rng)
module = LinearRegression()
params = module.init(key, X)['params']
model_params = params

y_predicted_initial = module.apply({'params': model_params}, X)
plt.plot(x, f(x), label='True function')
plt.scatter(X, y_predicted_initial, label='Initial prediction')  # Corrected line
plt.scatter(X, Y, label='Data')
plt.legend()
plt.show()

optimizer_def = optax.sgd(learning_rate=0.1, momentum=0.9)
optimizer_state = optimizer_def.init(model_params)
train_steps = 100

@jax.jit
def loss_fn(model_params):
    y_predicted = module.apply({'params': model_params}, X)
    return jnp.square(Y - y_predicted).mean()

@jax.jit
def train_step(optimizer_state, model_params):
    loss, grads = jax.value_and_grad(loss_fn)(model_params)
    updates, new_optimizer_state = optimizer_def.update(grads, optimizer_state, model_params)
    new_model_params = optax.apply_updates(model_params, updates)
    return new_optimizer_state, new_model_params, loss

for i in range(train_steps):
    optimizer_state, model_params, loss = train_step(optimizer_state, model_params)
print("MSE :", loss)

trained_model_params = model_params
print("Сургасан моделийн параметр :")
print(trained_model_params)
plt.plot(x, f(x), label='True function')
y_predicted_trained = module.apply({'params': trained_model_params}, x.reshape(-1, 1))
plt.plot(x, y_predicted_trained, label='Trained prediction')
plt.scatter(X, Y, label='Data')
plt.legend()
plt.show()  