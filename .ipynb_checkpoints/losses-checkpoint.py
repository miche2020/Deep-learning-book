{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40760633",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "\n",
    "def mse_loss(y_true, y_pred):\n",
    "    \"\"\"\n",
    "    Mean Squared Error loss using TF2\n",
    "    \"\"\"\n",
    "    return tf.reduce_sum(tf.square(y_true - y_pred))\n",
    "\n",
    "def cross_entropy_loss(y_true, y_pred):\n",
    "    \"\"\"\n",
    "    Cross entropy loss using TF2.\n",
    "    \n",
    "    y_true and y_pred - ohe vectors\n",
    "    \"\"\"\n",
    "    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log1p(y_pred), 1))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
