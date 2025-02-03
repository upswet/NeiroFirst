package neiro4;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Описывает одиночный нейрон
 */
public class Neiro implements Serializable {
    TypeEnum typeEnum; //тип нейрона
    FunEnum fun = FunEnum.NONE; //тип функции активации
    List<Link> inputs = new ArrayList<>(); //входные связи
    List<Link> outputs = new ArrayList<>(); //выходные связи


    float valueInput; //входящий сигнал - сумма сигналов полученным по всем входящим связям
    float valueOutput; //исходящий сигнал - результат применения функции активации к входящему сигналу
    float target; //значение правильного ответа для выходного нейрона для обчения
    float delta; //дельта. Участвует в корректировке весов в алгоритме обратного распространения ошибки


    public Neiro(TypeEnum typeEnum, FunEnum fun) {
        this.typeEnum = typeEnum;
        this.fun = fun;

        if (typeEnum.equals(TypeEnum.BIAS))
            valueInput = 1F;
        else
            valueInput = 0F;
    }

    public void setTarget(float target) {
        if (!typeEnum.equals(TypeEnum.OUTPUT))
            throw new RuntimeException("Нельзя помещать целевое значение в любой нейрон кроме выходного");
        this.target = target;
    }

    /**
     * Поместить значение в нейрон. Только для входных нейронов
     */
    public void setValue(float valueInput) {
        if (!typeEnum.equals(TypeEnum.INPUT))
            throw new RuntimeException("Нельзя помещать входные данные в любой нейрон кроме входного");
        this.valueInput = valueInput;
    }

    /**
     * Получить выходное значение нейрона
     */
    public float getValue() {
        return this.valueOutput;
    }

    /**
     * Цикл прямого распространения сигнала
     */
    public void forwardProcess() {
        this.valueOutput = this.fun(valueInput);
        this.outputs.forEach(link -> link.sendSignal(valueOutput));
        if (!this.typeEnum.equals(TypeEnum.BIAS))
            valueInput = 0F;//т.к. суммирование входов уже зкончилось ибо пришло время вычислять выход
    }


    /**
     * Цикл обратного распространения ошибки и коррекции весов исходящих связей нейрона
     *
     * @param lr     - коэффициент обучения
     * @param moment - момент обучения
     */
    public void backwardProcess(float lr, float moment) {
        switch (typeEnum) {
            case OUTPUT -> {
                //подсчитаем дельту-ошибки
                float err = target - getValue();
                delta = err * derivative(valueOutput);
            }
            case HIDDEN -> {
                //подсчитаем дельту-ошибки
                //откорректируем веса исходящих связей

                delta = 0F;
                for (Link link : outputs) {
                    Neiro neiroOutput = link.output;
                    delta += neiroOutput.delta * link.weight;

                    link.weightCorrect(lr, moment);
                }
                delta = derivative(valueOutput) * delta;
            }
            case INPUT, BIAS -> {
                //откоректируем веса исходящих связей
                for (Link link : outputs)
                    link.weightCorrect(lr, moment);
            }
        }

    }


    /**
     * Событие получения нейроном сигнала по одной из входящих связей при прямом проходе
     */
    public void onRecive(float value) {
        valueInput += value;
    }


    /**
     * функция активации
     */
    public float fun(float v) {
        switch (fun) {
            case SIGMOID -> {
                return (float) (1.0 / (1 + Math.exp(-v)));
            }
            default -> {
                return v;
            }
        }
    }

    /**
     * Производная функции активации
     */
    public float derivative(float v) {
        switch (fun) {
            case SIGMOID -> {
                return v * (1 - v);
            }
            default -> {
                return v;
            }
        }
    }
}
