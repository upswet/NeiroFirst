package neiro3_virual_process;

import java.io.Serializable;
import java.util.Random;

/**Описывает связь между нейронами*/
public class Link implements Serializable {
    static Random randomizer=new Random();

    Neiro input; //входной нейрон
    Neiro output;//выходной нейрон
    float weight;//вес связи
    float wd=0F;//предыдущая корректировка веса данной связи (необходимо для использоввания момента).
    float grad;//градиент, то есть направление в котором нам следует корректировать вес связи

    public Link(Neiro input, Neiro output, float weight) {
        this.input = input;
        this.output = output;
        this.weight = weight;

        this.input.outputs.add(this);
        this.output.inputs.add(this);
    }

    /**Генерация случайного веса связи по умолчанию*/
    public static float generateWeight(){
        return randomizer.nextFloat() - 0.5F;
    }

    /**Корректировка веса связи на основе исходящего значения входного нейрона, дельты выходного и таких параметров как
     * @param lr - коэффициент обучения
     * @param moment - момент обучения (исп в методе момента который здесь применяется)*/
    public void weightCorrect(float lr, float moment){
        grad=input.getValue()*output.delta;
        wd=lr*grad+moment*wd;
        weight+=wd;
    }

    /**Разорвать данную связь*/
    public void terminate(){
        this.input.outputs.remove(this);
        this.output.inputs.remove(this);
    }

    /**Связь передаёт сигнал выданный нейроном-отправителем в нейрон-поулчатель */
    public void sendSignal(float value){
        this.output.onRecive(value*this.weight);
    }

    /**Связь генерирует событие для нейрона-отправителя что нейрон-получатель вычислил свою дельту*/
    public void checkDelta(){
        this.input.onOutputDelteCalc();
    }
}
