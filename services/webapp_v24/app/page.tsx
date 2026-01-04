import TerminalWidget from '../components/TerminalWidget';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-slate-900 text-white">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
          Adam v24.0&nbsp;
          <code className="font-mono font-bold">Pragmatic Edition</code>
        </p>
      </div>

      <div className="w-full max-w-4xl grid grid-cols-1 gap-4">
        <div className="p-6 border border-slate-700 rounded-xl bg-slate-800/50">
          <h2 className="text-xl font-bold mb-4">Market Operations</h2>
          <TerminalWidget />
        </div>

        <div className="grid grid-cols-2 gap-4">
           <div className="p-6 border border-slate-700 rounded-xl bg-slate-800/50">
             <h2 className="text-xl font-bold mb-2">Live Orders</h2>
             <div className="h-32 bg-slate-900/50 rounded flex items-center justify-center text-slate-500">
               No active orders
             </div>
           </div>
           <div className="p-6 border border-slate-700 rounded-xl bg-slate-800/50">
             <h2 className="text-xl font-bold mb-2">System Health</h2>
             <div className="h-32 bg-slate-900/50 rounded flex items-center justify-center text-green-500">
               Nominal
             </div>
           </div>
        </div>
      </div>
    </main>
  );
}
