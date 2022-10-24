/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"
#include "weight.h"


static const int featureSize = 4;
static const int featureNum = 8;
static const long long tilesNum = pow(2,15);

const std::array<std::array<int, featureSize> ,featureNum> feature = {{
	{{0,1,2,3}},

	{{4,5,6,7}},
		
	{{8,9,10,11}},

	{{12,13,14,15}},

	{{0,4,8,12}},

	{{1,5,9,13}},

	{{2,6,10,14}},

	{{3,7,11,15}},
}};

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

class greedy1step_slider : public random_agent {
public:
	greedy1step_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		int bestOp = -1;
		board::reward bestReward = -1;
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if(reward > bestReward){
				bestReward = reward;
				bestOp = op;
			}
		}
		if (bestReward != -1) return action::slide(bestOp);
		else return action();
	}

private:
	std::array<int, 4> opcode;
};

class greedy2step_slider : public random_agent {
public:
	greedy2step_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		int bestOp = -1;
		board::reward bestReward = -1;
		for (int op : opcode) {
			auto firstBoard = board(before);
			board::reward reward1 = firstBoard.slide(op);
			board::reward bestReward2 = -1;
			for (int op : opcode) {
				board::reward reward2 = board(firstBoard).slide(op);
				if( reward2 > bestReward2 ) bestReward2 = reward2;
			}
			if ( reward1 + bestReward2 > bestReward ) {
				bestReward = reward1 + bestReward2;
				bestOp = op;
			}
		}
		if (bestReward != -1) return action::slide(bestOp);
		else return action();
	}

private:
	std::array<int, 4> opcode;
};

class greedy3step_slider : public random_agent {
public:
	greedy3step_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		int bestOp = -1;
		board::reward bestReward = -1;
		for (int op : opcode) {
			auto firstBoard = board(before);
			board::reward reward1 = firstBoard.slide(op);
			board::reward bestReward2 = -1;
			for (int op : opcode) {
				auto secondBoard = board(firstBoard);
				board::reward reward2 = secondBoard.slide(op);
				board::reward bestReward3 = -1;
				for (int op : opcode) {
					board::reward reward3 = board(secondBoard).slide(op);
					if(reward3 > bestReward3) bestReward3 = reward3;
				}
				if ( reward2 + bestReward3 > bestReward2 ) bestReward2 = reward2 + bestReward3;
			}
			if ( reward1 + bestReward2 > bestReward ) {
				bestReward = reward1 + bestReward2;
				bestOp = op;
			}
		}
		if (bestReward != -1) return action::slide(bestOp);
		else return action();
	}

private:
	std::array<int, 4> opcode;
};

class tdLearning_slider: public weight_agent {
public:
	tdLearning_slider(const std::string& args = "") : weight_agent("name=slide role=slider " + args),
	opcode({ 0, 1, 2, 3 }){
		// create 8 tuples in the net
		for (int i = 0; i < featureNum; i++) {
			net.emplace_back(weight(tilesNum));
		}
	}

	virtual action take_action(const board& before) {
		//Feature = feature;
		auto b = board(before);
		int bestop = SelectBestOp(before);
		int bestReward = b.slide(bestop);

		if (bestop != -1) {
			switch (round) {
				case 0:
					prev = b;
					round++;
					break;
				case 1:
					next = b;
					train(prev,next,bestReward);
					round++;
					break;
				default:
					prev = next;
					next = b;
					train(prev,next,bestReward);
					break;
			}
			return action::slide(bestop);
		}
		return action();
	}

	int FindTileIndex(const board& b, int featureIndex) {
		long long tileIndex = 0;
		for (int i = 0; i < featureSize; i++) {
			int row = feature[featureIndex][i] / 4;
			int column = feature[featureIndex][i] % 4;
			tileIndex += b[row][column];
		}
	}

	float CalculateBoardValue(const board& before) {
		float value = 0;
		auto b = board(before);
		for (int ind = 0; ind < featureNum; ind++) {
			int tileIndex = FindTileIndex(b, ind);
			value += net[ind][tileIndex];
		}
		return value;
	}

	int SelectBestOp(const board& before) {
		int bestop = -1;
		float maxValue = -std::numeric_limits<float>::infinity();
		
		for (int op : opcode) {
			auto after = board(before);
			board::reward reward = after.slide(op);
			float boardValue = CalculateBoardValue(after);
			if (reward + boardValue > maxValue) {
				bestop = op;
				maxValue = reward + boardValue;
			}
		}
		return bestop;
	}

	void train(const board& prev, const board& next, int reward) {
		float learningRate = 0.1/32;
		float delta = CalculateBoardValue(next) - CalculateBoardValue(prev) + reward ;
		tdError += delta*learningRate;	
		double v_s = (reward==-1) ? 0 : learningRate * delta;

		for (int ind = 0; ind < featureNum; ind++) {
			int tileIndex = FindTileIndex(prev, ind);
			net[ind][tileIndex] += tdError;
		}
	}

private:
	std::array<int, 4> opcode;
	//std::array<std::array<int, featureSize>, featureNum> Feature;
	int round = 0;
	board prev, next;
	float tdError = 0;
};