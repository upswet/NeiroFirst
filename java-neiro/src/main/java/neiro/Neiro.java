package neiro;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**Описывает одиночный нейрон*/
public class Neiro implements Serializable {
    TypeEnum typeEnum; //тип нейрона
    FunEnum fun= FunEnum.NONE; //тип функции активации
    List<Link> inputs = new ArrayList<>(); //входные связи
    List<Link> outputs = new ArrayList<>(); //выходные связи

    float valueInput; //входящий сигнал - сумма сигналов полученным по всем входящим связям
    float valueOutput; //исходящий сигнал - результат применения функции активации к входящему сигналу
    float delta; //делтьа. Участвует в корректировке весов в алгоритме обратного распространения ошибки


    public Neiro(TypeEnum typeEnum, FunEnum fun) {
        this.typeEnum = typeEnum;
        this.fun=fun;

        if (typeEnum.equals(TypeEnum.BIAS))
            valueInput=1F;
        else
            valueInput=0F;
    }

    /**Поместить значение в нейрон. Только для входных нейронов*/
    public void setValue(float valueInput){
        if (!typeEnum.equals(TypeEnum.INPUT)) throw new RuntimeException("Нельзя помещать значение в любой нейрон кроме входного");
        this.valueInput=valueInput;
    }
    /**Получить выходное значение нейрона*/
    public float getValue(){
        return this.valueOutput;
    }
    /**Преобразование входного значения в выходное с обнулением входного так как оно больше нам не нужно
     * Запускается когда получены все сигналы по входным связям (или, что аналогично, все нейроны предыдущих слоёв уже отправили свои выходные значения по своим выхоным связям*/
    public void forward(){
        valueOutput=fun(valueInput);
        valueInput=0F;//т.к. суммирование входов уже зкончилось ибо пришло время вычислять выход
    }

    /**Получение нейроном сигнала по одной из входящих связей*/
    public void recive(float value){
        valueInput+=value;
    }

    /**функция активации*/
    public float fun(float v){
        switch (fun){
            case SIGMOID -> {return (float) (1.0 / (1 + Math.exp(-v)));}
            default -> {return v;}
        }
    }

    /**Производная функции активации*/
    public float derivative(float v){
        switch (fun){
            case SIGMOID -> {return v * (1 - v);}
            default -> {return v;}
        }
    }
}
