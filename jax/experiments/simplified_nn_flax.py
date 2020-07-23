import flax
import jax
from jax import numpy as jnp
import numpy as np

# Flax сурах дадлага : https://flax.readthedocs.io/en/latest/notebooks/flax_guided_tour.html


# Flax ашиглан хамгийн хялбар неорон давхарга тодорхойлох
class Dense(flax.nn.Module):
    def apply(self,x,
            features,
            kernel_initialization = jax.nn.initializers.lecun_normal(),
            bias_initialization   = jax.nn.initializers.zeros
            ):
        input_features = x.shape[-1]
        kernel_shape   = (input_features, features)
        kernel         = self.param('kernel', kernel_shape, kernel_initialization)
        bias           = self.param('bias'  , (features,) , bias_initialization  )
        return jnp.dot(x, kernel) + bias


# Неорон сүлжээний оролтын өгөгдлийг бэлтгэх
# энд 1 ширхэг дата буюу batch нь 1 гэсэн үг 4-н оролтын өгөгдлийн дүрс
x = jnp.ones((1, 4))

# Оролтын өгөгдлийн дүрсээс хамааруулан неорон сүлжээний жингүүдийг 
# эхний байдлаар цэнэглэн гаргаж авах
y, params = Dense.init(jax.random.PRNGKey(0), x, features=4)
print("Неорон сүлжээний эхний цэнэглэгдсэн параметрүүд :")
print(params)
print("Эхний байдлаар predict хийсэн гаралт :")
print(y)

# Цэнэглэгдсэн параметрүүдээ хэрэглэн энгийн predict хийх forward дуудалт
prediction = Dense.call(params, x, features=4)
print("predict хийсэн дуудалт :")
print(prediction)


# Зөвхөн hyperparameter-үүд тохируулсан модуль үүсгэе гэвэл partial функц ашиглаж болно
module    = Dense.partial(features=4)
# Дараа нь жингүүдээ цэнэглэж болно
_, params = module.init(jax.random.PRNGKey(0), x)
print("Модуль үүсгэсний дараа цэнэглэсэн жингүүд :")
print(params)
# Модуль хэрэглэн forward дуудалт хийх
prediction = module.call(params, x)
print("Модулиас хийсэн prediction дуудалтын үр дүн :")
print(prediction)


# Модуль ашигласнаар хооронд нь хольж бүрдүүлж болдог нарийн 
# бүтэцтэй неорон сүлжээ зохион байгуулах боломж бүрддэг

def relu(x):
    return jnp.maximum(0., x)

# Multi Layer Perceptron буюу олон давхаргатай хялбар неорон сүлжээ
class MLP(flax.nn.Module):
    def apply(self, x,
            hidden_features,
            output_features,
            activation_fn):
        z = Dense(x, hidden_features)
        h = activation_fn(z)
        y = Dense(h, output_features)
        return y

module    = MLP.partial(hidden_features=8, output_features=4, activation_fn=relu)
y, params = module.init(jax.random.PRNGKey(0), x)
# Нарийн бүтэцтэй олон давхаргатай неорон сүлжээ бий болгосон учраас 
# параметрүүд нь шаталсан бүтэцтэй dictionary байна
print("MLP параметрүүд :")
print(params)
print("MLP predict хийсэн утгууд :")
print(y)


# Хэрвээ неорон давхаргад нэр оноогоогүй байвал автоматаар дугаарлагдаад явна
# тогтсон нэр өгч ашиглах боломжтой

class NamedMLP(flax.nn.Module):
    def apply(self, x,
            hidden_features,
            output_features,
            activation_fn):
        z = Dense(x, hidden_features, name='hidden')
        h = activation_fn(z)
        y = Dense(h, output_features, name='out')
        return y

module    = NamedMLP.partial(hidden_features=8, output_features=4, activation_fn=relu)
_, params = module.init(jax.random.PRNGKey(0), x)

print("Нэрлэсэн давхаргууд бүхий неорон сүлжээний бүтэц :")
print(jax.tree_map(np.shape, params))


# Олон call дуудалтанд дундын parameter-рүүд хэрэглэж болно

class SimpleRNN(flax.nn.Module):
    def apply(self, x, iterations=3):
        dense = Dense.shared(features=x.shape[-1],
                kernel_initialization=jax.nn.initializers.orthogonal(), name='cell')
        ys = []
        for i in range(iterations):
            x = dense(x)
            ys.append(x)
        return ys

ys, params = SimpleRNN.init(jax.random.PRNGKey(0), x)
print("Хялбар Recurrent Neural Network-ийн гаралт :")
print(ys)

# cell давхаргыг 3-н удаа дуудсанч shared параметер горимоор үүсгээн тул нэмэлтээр 
# 3-н давхарга болж орж ирэхгүй нэг л давхарга байдлаар харагдана гэсэн үг
# {'cell': {'bias': (4,), 'kernel': (4, 4)}}
print("Recurrent Neural Network-ийн параметер бүтэц :")
print(jax.tree_map(np.shape, params))


# Оролтын өгөгдлийн дүрснээс хамаарч модулийг init хийх нь заримдаа шаардлагагүй нэмэлт 
# үйлдлүүд хийгдэх магадлалтай, зөвхөн оролтын дүрс нь ямар вэ гэдгээр init хийж болно
# {'cell': {'bias': (2,), 'kernel': (2, 2)}}
input_spec       = [(1,2)]
out_spec, params = SimpleRNN.init_by_shape(jax.random.PRNGKey(0), input_spec)
print("Shape inference горимоор барьсан неорон сүлжээний бүтэц :")
print(jax.tree_map(np.shape, params))


# Модуль хэрэглэснээр неорон сүлжээний параметрүүдийг init болон call функцын тусламжтайгаар
# ил хэлбэрээр track хийх боломж олгодог. Model класс бол Module классын багахан өргөтсөн
# хэлбэр бөгөөд заавал параметрүүдийг нь track хийж байн байн дамжуулаад байх шаардлагагүй
x      = jnp.ones((1,2))
module = Dense.partial(features=4)
ys, initial_params = module.init(jax.random.PRNGKey(0), x)
model  = flax.nn.Model(module, initial_params)
print("Моделийн бүтцийг харах :")
print(jax.tree_map(np.shape, model.params))

prediction = model(x)
print("Моделиор predict хийсэн үр дүн :")
print(prediction)

print("Моделийн параметрүүд :")
print(model.params)


# Моделийн параметрүүдийг нь шинэчилж сольж болно
biased_model = model.replace(
        params={
            'kernel': model.params['kernel'],
            'bias'  : model.params['bias'  ]+1.
            })

print("Параметрүүдийг нь шинэчилсэн моделийн параметрүүд :")
print(biased_model.params)


# Модель нь JAX-ны pytree объект байдлаар зохион байгуулагддаг тул 
# JAX-н native хувиргалтуудад асуудалгүйгээр хэрэглэх боломжтой

# Жишээлбэл модель объект хэрэглэн градиент утгууд олж болно
def loss_fn(model):
    y = model(x)
    return jnp.mean(jnp.square(y))

model_gradients = jax.grad(loss_fn)(model)
print("Моделийн loss-оос авсан gradient утгууд")
print(model_gradients.params)
